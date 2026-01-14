import pandas as pd
import re

def build_edit_pattern():
    phrases = [
        r'edit(?:\s*\d+)?',
        r'eta',
        r'update',
        r'final update',
        r'quick update',
        r'small edit',
        r'minor update',
        r'clarification',
        r'correction',
        r'note',
        r'side note',
    ]
    return r'(^|\n{2,})\s*(' + '|'.join(phrases) + r')\b[\s:–—-]+'

EDIT_REGEX = re.compile(build_edit_pattern(), flags=re.IGNORECASE | re.DOTALL)


def clean_submission_text(text):
    """
    Removes Reddit-style edit/update/tldr sections from the start or end of a post.
    Preserves the main content between.
    """
    if not isinstance(text, str):
        return text

    # Normalize line breaks
    text = text.replace('\r\n', '\n').replace('\r', '\n').strip()

    # === Remove top paragraph if it's an edit/update ===
    top_match = re.match(r'^\s*(' + build_edit_pattern() + r')', text, flags=re.IGNORECASE)
    if top_match:
        parts = re.split(r'\n{2,}', text, maxsplit=1)
        if len(parts) == 2:
            text = parts[1].strip()
        else:
            return ''  # Only edit content found

    # === Remove bottom paragraph starting with edit/update ===
    bottom_match = EDIT_REGEX.search(text)
    if bottom_match:
        text = text[:bottom_match.start()].strip()

    return text


def clean_submissions_df(df, text_column=None):
    """
    Clean submission texts in a pandas DataFrame.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the submissions
        text_column (str, optional): Name of the column containing the text. 
                                   If None, will look for 'text' or 'body'
    
    Returns:
        pandas.DataFrame: DataFrame with cleaned texts in a new column
    """
    if text_column is None:
        text_column = 'text' if 'text' in df.columns else 'body'
    
    if text_column not in df.columns:
        raise ValueError(f"Could not find text column in DataFrame. Available columns: {df.columns.tolist()}")
    
    # Create a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    # Add new column with cleaned text
    cleaned_column = f"{text_column}_cleaned"
    df_cleaned[cleaned_column] = df_cleaned[text_column].apply(clean_submission_text)
    
    return df_cleaned
