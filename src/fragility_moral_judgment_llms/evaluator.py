import anthropic
import logging
import json
import os
import re
import pickle

from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request
from openai import OpenAI
from google import genai
from google.genai import types
from tenacity import retry, wait_random_exponential, stop_after_attempt, before_sleep_log
from tqdm import tqdm

tenacity_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class Evaluator:
    def __init__(self, client, system_prompt, dilemmas, api_key=None):
        """Initialize the evaluator with API keys and setup models."""
        self.system_prompt = system_prompt
        # Ensure dilemmas is always a list
        if isinstance(dilemmas, str):
            self.dilemmas = [dilemmas]
        else:
            self.dilemmas = dilemmas

        # Setup client and query method based on provider
        if client == "openai":
            if api_key is None:
                api_key = os.getenv('OPENAI_API_KEY')
            self.client = OpenAI(api_key=api_key)
            self.query = self._query_openai_streaming
        elif client == "anthropic":
            if api_key is None:
                api_key = os.getenv('ANTHROPIC_API_KEY')
            self.client = anthropic.Client(api_key=api_key)
            self.query = self._query_anthropic_streaming
        elif client == "google":
            if api_key is None:
                api_key = os.getenv('GEMINI_API_KEY')
            self.client = genai.Client(api_key=api_key)
            self.query = self._query_google_streaming

    @staticmethod
    def extract_json(text: str) -> dict:
        """
        Extracts a JSON object from a string, attempting to clean common issues
        like markdown fences, escaped apostrophes, newlines within strings,
        and unescaped quotes within strings. (v3)

        Args:
            text: The input string potentially containing a JSON object.

        Returns:
            A dictionary parsed from the JSON object.

        Raises:
            ValueError: If a JSON object cannot be found or parsed.
        """
        text = text.strip()

        # 1. Look for ```json ... ``` or ``` ... ``` markdown blocks
        match = re.search(r'```(?:json)?\s*({.*?})\s*```', text, re.DOTALL)
        if match:
            text = match.group(1)
        else:
            # 2. If no markdown, find the outermost curly braces
            start_index = text.find('{')
            end_index   = text.rfind('}')

            if start_index != -1 and end_index != -1 and end_index > start_index:
                text = text[start_index:end_index + 1]
            elif start_index != -1 and end_index == -1:          # ← NEW: handle missing }
                text = text[start_index:]                        # keep everything to the end;
                # the brace-count fix below will append the } later
            else:
                raise ValueError("Could not find a JSON object in the text.")

        # 3. Fix illegal \' escape by replacing it with a standard apostrophe
        text = text.replace(r"\'", "'")

        # 4. Attempt to fix unescaped characters within strings.
        buf = []
        in_string = False
        i = 0
        text_len = len(text)
        while i < text_len:
            char = text[i]

            # Check if the current character is a quote and if it's escaped
            is_escaped = False
            if char == '"':
                k = i - 1
                num_backslashes = 0
                while k >= 0 and text[k] == '\\':
                    num_backslashes += 1
                    k -= 1
                is_escaped = (num_backslashes % 2 != 0)

            if char == '"' and not is_escaped:
                if not in_string:
                    # This is an opening quote
                    in_string = True
                    buf.append(char)
                else:
                    # This could be a closing quote or an internal quote.
                    # Peek ahead to see if it's followed by a delimiter.
                    is_closing = False
                    j = i + 1
                    # Skip whitespace characters
                    while j < text_len and text[j] in ' \t\n\r':
                        j += 1
                    
                    # Check if it's the end OR followed by a valid delimiter (:, ,, }, ])
                    # *** ADDED ':' to this check ***
                    if j == text_len or text[j] in ',}:]':
                        is_closing = True

                    if is_closing:
                        # It's a closing quote
                        in_string = False
                        buf.append(char)
                    else:
                        # It's an internal quote, escape it.
                        buf.append('\\"')
            elif char == '\n' and in_string:
                buf.append('\\n')
            elif char == '\t' and in_string:
                buf.append('\\t')
            else:
                buf.append(char)
                
            i += 1
        text = "".join(buf)

        # 5. Handle potential missing closing brace (simple check)
        if text.count('{') > text.count('}'):
            text += '}'

        # 6. Try to parse the cleaned JSON string
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON after cleaning: {e}\nProcessed text: '{text}'") from e

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(tenacity_logger, logging.INFO)
    )
    def _query_openai_streaming(self, model, dilemma, temperature=0.4, seed=None):
        """
        Query the OpenAI API with a system prompt and dilemma, using streaming.
        Retries on failure with exponential backoff.
        """
        response = self.client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": dilemma}
            ],
            temperature=temperature,
            seed=seed
        )
        output = self.extract_json(response.choices[0].message.content)
        return output

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(tenacity_logger, logging.INFO)
    )
    def _query_anthropic_streaming(self, model, dilemma, temperature=0.4):
        """
        Query the Anthropic Claude API with a system prompt and dilemma.
        Retries on failure with exponential backoff.
        """
        response = self.client.messages.create(
            model=model,
            system=self.system_prompt,
            messages=[{"role": "user", "content": dilemma}],
            temperature=temperature,
            max_tokens=1024)
        output = self.extract_json(response.content[0].text)
        return output

    @retry(
        wait=wait_random_exponential(min=1, max=60),
        stop=stop_after_attempt(5),
        before_sleep=before_sleep_log(tenacity_logger, logging.INFO)
    )
    def _query_google_streaming(self, model, dilemma, temperature=0.4, max_tokens=1024):
        """
        Queries the Google Gemini API with a given dilemma in a streaming fashion.

        This function sends a single dilemma to the specified Google Gemini model
        along with a system prompt and configuration parameters. It includes retry logic
        with exponential backoff for robustness against transient failures.

        Args:
            model (str): The Google Gemini model to use (e.g., "gemini-pro").
            dilemma (str): The specific dilemma text to be evaluated by the model.
            temperature (float, optional): Controls the randomness of the output.
                                           Lower values mean less random. Defaults to 0.4.
            max_tokens (int, optional): The maximum number of tokens to generate in the
                                        model's response. Defaults to 1024.

        Returns:
            dict: The extracted JSON output from the model's response.

        Raises:
            tenacity.RetryError: If the function fails after all retry attempts.
            Exception: Any other exception raised by the `self.client.models.generate_content`
                       call or `self.extract_json` if not caught by retry.
        """
        response = self.client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(
                system_instruction=self.system_prompt,
                max_output_tokens=max_tokens,
                temperature=temperature),
            contents=dilemma)
        if response.prompt_feedback is not None:
            if response.prompt_feedback.block_reason == 'PROHIBITED_CONTENT':
                output = {'judgment': 'REFUSAL', 'explanation': 'Refusal due to Prohibited Content.'}
        else:
            output = self.extract_json(response.text)
        return output

    def write_openai_batch_jsonl(
        self, model, path, exp_id=None, temperature=0.4, seed=None
    ):
        """
        Writes a JSONL file formatted for OpenAI's batch API.

        Each line in the file represents a request for evaluating a dilemma,
        including model, system prompt, user message, and temperature.

        Args:
            model (str): The OpenAI model to use (e.g., "gpt-4").
            path (str): The file path where the JSONL content will be written.
            exp_id (str, optional): An experiment ID to prefix custom IDs. Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.4.
            seed (int, optional): A seed for reproducible sampling. Defaults to None.

        Returns:
            str: The path to the created JSONL file.
        """
        # Format the experiment ID for use in custom_id
        if exp_id is None:
            exp_id = ""
        else:
            exp_id = f"exp{exp_id}_"

        # Open the specified file in write mode
        with open(path, "w") as file:
            # Iterate through each dilemma to create a batch request payload
            for idx, dilemma in enumerate(self.dilemmas):
                # Construct the payload for a single OpenAI chat completion request
                payload = {
                    "custom_id": f"{exp_id}dilemma_{idx}",  # Unique ID for this request
                    "method": "POST",
                    "url": "/v1/chat/completions",
                    "body": {
                        "model": model,
                        "messages": [
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": dilemma}
                        ],
                        "temperature": temperature
                    }
                }
                # Add seed to the payload body if provided
                if seed is not None:
                    payload["body"]["seed"] = seed

                # Write the JSON payload as a single line followed by a newline
                file.write(json.dumps(payload) + "\n")

        # Return the path to the generated file
        return path

    def _query_openai_batch(self, model, path, exp_id=None, temperature=0.4, seed=None):
        """
        Prepares a JSONL file and submits it as a batch job to the OpenAI API.

        First, it generates a JSONL file containing individual chat completion requests.
        Then, it uploads this file to OpenAI and creates a new batch processing job.

        Args:
            model (str): The OpenAI model to use for the batch (e.g., "gpt-4").
            path (str): The local file path for the temporary JSONL batch input.
            exp_id (str, optional): An experiment ID for custom request IDs. Defaults to None.
            temperature (float, optional): The sampling temperature. Defaults to 0.4.
            seed (int, optional): A seed for reproducible sampling. Defaults to None.

        Returns:
            object: The OpenAI batch object, representing the submitted job.
        """
        # Generate the JSONL file containing all individual requests for the batch
        jsonl_file_path = self.write_openai_batch_jsonl(
            model=model,
            path=path,
            exp_id=exp_id,
            temperature=temperature,
            seed=seed
        )

        # Upload the generated JSONL file to OpenAI's file storage
        batch_input_file = self.client.files.create(
            file=open(jsonl_file_path, "rb"),
            purpose="batch"
        )

        # Create a new batch job using the uploaded file's ID.
        batch = self.client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h")

        return batch

    def _query_anthropic_batch(self, model, exp_id=None, temperature=0.4, max_tokens=1024):
        """
        Creates and sends a batch evaluation request to the Anthropic API.

        Builds individual message creation requests for each dilemma in `self.dilemmas`,
        then dispatches them as a single batch.

        Args:
            model (str): Anthropic model name (e.g., "claude-3-opus").
            exp_id (str, optional): Experiment ID for custom request IDs. Defaults to None.
            temperature (float, optional): Sampling temperature. Defaults to 0.4.
            seed (int, optional): Seed for reproducibility. Defaults to None.

        Returns:
            object: The batch response object from the Anthropic API.
        """
        requests = [
            Request(
                custom_id=f"{exp_id}dilemma_{idx}",
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    system=self.system_prompt,
                    max_tokens=max_tokens,
                    messages=[{"role": "user", "content": dilemma}],
                    temperature=temperature)
                )
            for idx, dilemma in enumerate(self.dilemmas)
        ]
        batch = self.client.messages.batches.create(requests=requests)
        return batch

    def evaluate_dilemmas_streaming(self, model, tracking=None, **kwargs):
        """
        Evaluates dilemmas one by one using the specified model.

        Iterates through `self.dilemmas`, querying each individually with `self.query`.
        Shows progress with `tqdm`.

        Args:
            model (str): The model name (e.g., "gpt-4").
            **kwargs: Additional arguments passed to `self.query`.

        Returns:
            list: Outputs from each dilemma's evaluation.
        """
        if tracking is not None and os.path.exists(tracking):
            with open(tracking, 'rb') as file:
                outputs = pickle.load(file)
        else:
            outputs = []

        n_outputs = len(outputs)
        dilemmas = self.dilemmas.iloc[n_outputs:]

        # Process each dilemma individually
        for idx, dilemma in tqdm(enumerate(dilemmas), total=len(dilemmas)):
            output = self.query(model=model,
                                dilemma=dilemma,
                                **kwargs)
            outputs.append(output)
            if tracking is not None:
                with open(tracking, 'wb') as file:
                    pickle.dump(outputs, file)

        return outputs

    def evaluate_dilemmas_batch(self, client, model, **kwargs):
        """
        Evaluates a batch of dilemmas using the specified client and model.

        This function iterates through a collection of dilemmas and queries the
        chosen language model client (OpenAI or Anthropic) to get outputs for
        each dilemma.

        Args:
            client (str): The name of the client to use for evaluation.
                Currently supported: "openai", "anthropic".
            model (str): The specific model to use from the chosen client (e.g.,
            "gpt-4", "claude-3"). 
            **kwargs: Additional keyword arguments to pass to the underlying
            client query function.

        Returns:
            list: A list of outputs returned by the chosen language model client
            for each dilemma in the batch. The structure of these outputs depends on the
            client's API response.

        Raises:
            ValueError: If an unsupported client is provided.
        """
        # Route the batch evaluation based on the specified client.
        if client == "openai":
            # Call the internal method to query OpenAI in a batch.
            # Pass the model and any additional keyword arguments (e.g., 'path').
            batch = self._query_openai_batch(
                model=model,
                path=kwargs.get('path'),  # Safely get 'path' from kwargs
                **kwargs)
        elif client == "anthropic":
            # Call the internal method to query Anthropic in a batch.
            # Anthropic queries might have different specific parameters.
            batch = self._query_anthropic_batch(
                model=model,
                **kwargs)
        else:
            # Raise an error if an unsupported client is specified.
            raise ValueError(f"Unsupported client: {client}. Please choose 'openai' or 'anthropic'.")

        # Return the collected batch outputs.
        return batch