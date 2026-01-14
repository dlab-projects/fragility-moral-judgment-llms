import requests
import time


def get_access_token(client_id, client_secret):
    """
    Obtains an OAuth2 access token for the Reddit API.

    Args:
        client_id (str): The client ID of the Reddit application.
        client_secret (str): The client secret of the Reddit application.

    Returns:
        str: The access token to be used for authenticated API requests.

    Raises:
        requests.exceptions.HTTPError: If the request to obtain the access token fails.
    """
    # Set up the headers for the API request, including the access token for authentication
    auth = requests.auth.HTTPBasicAuth(client_id, client_secret)
    data = {'grant_type': 'client_credentials'}
    headers = {'User-Agent': 'MyRedditApp/0.1'}
    
    res = requests.post('https://www.reddit.com/api/v1/access_token',
                        auth=auth, data=data, headers=headers)
    res.raise_for_status()
    return res.json()['access_token']


def get_top_comments(submission_id, access_token, max_retries=3, delay=60):
    """
    Fetches the top-level comments from a Reddit submission.

    Args:
        submission_id (str): The ID of the Reddit submission to fetch comments from.
        access_token (str): The OAuth2 access token for authenticating the API request.
        max_retries (int): Number of retry attempts on failure or rate limiting.
        delay (float): Delay (in seconds) between retries to respect rate limits.

    Returns:
        dict: The raw JSON response from the Reddit API containing the submission's comments.

    Raises:
        Exception: If all retries fail or an unexpected error occurs.

    Note:
        This function currently returns the entire JSON response instead of extracting and returning
        only the comments. The comments can be accessed from the response data.
    """
    headers = {
        "Authorization": f"bearer {access_token}",
        "User-Agent": "MyApp/0.0.1"
    }
    url = f"https://oauth.reddit.com/comments/{submission_id}?limit=100&depth=1&sort=top"

    # Iterate over retry attempts
    for attempt in range(max_retries):
        response = requests.get(url, headers=headers)
        # Successful response
        if response.status_code == 200:
            return response.json()
        # Rate limit response
        elif response.status_code == 429:
            print(f"Rate limit hit (429). Retrying in {delay}s...")
        # Other Error
        else:
            print(f"Error {response.status_code} on attempt {attempt+1}: {response.text}")
        time.sleep(delay)

    raise Exception(f"Failed to fetch comments for submission {submission_id} after {max_retries} attempts.")
