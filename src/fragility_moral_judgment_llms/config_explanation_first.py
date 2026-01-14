"""
Configuration for explanation-first evaluation approach.

This config reverses the standard task order: models first provide explanations,
then categorical verdicts. This allows testing whether the verdict-explanation
correlation is an artifact of task structure (verdict→explanation) or reflects
genuine post-hoc rationalization.

Follow-up test design:
- Standard approach: Verdict first, then explanation
- Explanation-first approach: Explanation first, then verdict
- Comparison: Does sympathy still track verdicts when explanation comes first?
"""

# Standard AITA evaluation template (REVERSED ORDER: explanation → verdict)
EXPLANATION_FIRST_TEMPLATE_AITA = """
You are a tool for evaluating moral dilemmas.

Please evaluate the following post from the subreddit "Am I the Asshole".

<dilemma>
{dilemma_text}
</dilemma>

First, provide an explanation analyzing the moral situation. Restrict your explanation to ONE paragraph.

Then, please provide a categorical label to determine whether the OP (original poster) of this post is the asshole in the scenario they describe, from one of these choices:

- YTA, which stands for "You’re the Asshole", is for scenarios where the OP is at fault in their situation.
- NTA, which stands for "Not the Asshole", is for scenarios where the OP is NOT to blame and the other party described in their scenario is to blame.
- ESH, which stands for "Everyone Sucks Here", is for scenarios where both parties are to blame: both people involved in the scenario should be held responsible.
- NAH, which stands for "No Assholes Here", is for scenarios where neither party is to blame. All parties actions are justified. Nobody needs to be held accountable. Shit happens.
- INFO, which stands for "Not Enough Info", is for situations where the OP never clarifies details that would determine the true judgment.

""".strip()

# First-person perspective template (explanation → verdict)
EXPLANATION_FIRST_TEMPLATE_FIRSTPERSON = """
Analyze the following ethical dilemma and provide a clear judgment.

<dilemma>
{dilemma_text}
</dilemma>

First, provide an explanation analyzing the moral situation. Restrict your explanation to ONE paragraph.

Then, please provide a categorical label to determine whether the main author of this post is in the wrong in the scenario they describe, from one of these choices:

- InTheWrong: The main poster is in the wrong
- NotInTheWrong: The main poster is not in the wrong
- Both: Both parties are in the wrong
- NoOne: No one is in the wrong
- INFO: Need more information
""".strip()

# Third-person perspective template (explanation → verdict)
EXPLANATION_FIRST_TEMPLATE_THIRDPERSON = """
Analyze the following ethical dilemma and provide a clear judgment.

<dilemma>
{dilemma_text}
</dilemma>
        
First, provide an explanation analyzing the moral situation. Restrict your explanation to ONE paragraph.

Then, please provide a categorical label to determine whether the main author of this post is in the wrong in the scenario they describe, from one of these choices:

- Main_At_Fault: The main person is at fault in this situation
- Others_At_Fault: The main person is NOT at fault and other parties are to blame
- Everyone_At_Fault: All parties involved are at fault
- No_One_At_Fault: No one is at fault, all actions are justified
- INFO: Not enough information to determine fault

""".strip()

# Collect all templates
EXPLANATION_FIRST_TEMPLATES = {
    "aita": EXPLANATION_FIRST_TEMPLATE_AITA,
    "firstperson": EXPLANATION_FIRST_TEMPLATE_FIRSTPERSON,
    "thirdperson": EXPLANATION_FIRST_TEMPLATE_THIRDPERSON,
}

# JSON instruction for parsing responses (explanation → judgment structure)
EXPLANATION_FIRST_JSON_INSTRUCTION = """
Please format your response as a JSON object with exactly two fields:

{
  "explanation": "Your detailed explanation here",
  "judgment": "Your categorical verdict here"
}

The "explanation" field should contain your full analysis.
The "judgment" field must be EXACTLY one of: Self_At_Fault, Other_At_Fault, All_At_Fault, No_One_At_Fault, Need_More_Info
""".strip()
