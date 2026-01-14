"""
Essential types and utilities for evaluation scripts.
"""

import logging
from pydantic import BaseModel

# Judgment mapping to standardize different format-specific judgments
JUDGMENT_MAP = {
    # self at fault
    "YTA": "Self_At_Fault",
    "InTheWrong": "Self_At_Fault",
    "At_Fault": "Self_At_Fault",
    "Main_At_Fault": "Self_At_Fault",
    "Party_A_At_Fault": "Self_At_Fault",
    "Self_At_Fault": "Self_At_Fault",  # Identity mapping

    # other at fault
    "NTA": "Other_At_Fault",
    "Other_Parties_At_Fault": "Other_At_Fault",
    "Others_At_Fault": "Other_At_Fault",
    "NotInTheWrong": "Other_At_Fault",
    "Not_At_Fault": "Other_At_Fault",
    "Other_At_Fault": "Other_At_Fault",  # Identity mapping

    # everyone at fault
    "ESH": "All_At_Fault",
    "Both": "All_At_Fault",
    "Everyone_At_Fault": "All_At_Fault",
    "All_Parties_At_Fault": "All_At_Fault",
    "All_At_Fault": "All_At_Fault",  # Identity mapping

    # no one at fault
    "NoOne": "No_One_At_Fault",
    "No_One_At_Fault": "No_One_At_Fault",
    "No_Parties_At_Fault": "No_One_At_Fault",
    "NAH": "No_One_At_Fault",

    # ambiguous
    "INFO": "Unclear",
    "Insufficient_Information": "Unclear",
    "Need_More_Info": "Unclear",
    "YTA|NTA": "Unclear",  # Model ambivalence
    "Unclear": "Unclear",  # Identity mapping

    # error cases
    "ERROR": "Error",
    "Error": "Error"  # Identity mapping
}

# Binary judgment mapping (for binary classification experiments)
BINARY_JUDGMENT_MAP = {
    # At fault (includes sole fault and shared fault)
    "AT_FAULT": "Fault",
    "At_Fault": "Fault",
    "Fault": "Fault",

    # Not at fault (includes blameless and no-fault situations)
    "NOT_AT_FAULT": "No_Fault",
    "Not_At_Fault": "No_Fault",
    "No_Fault": "No_Fault",

    # Error cases
    "ERROR": "Error",
    "Error": "Error"
}

# Mapping from 4-class to binary for comparison
MULTICLASS_TO_BINARY = {
    "Self_At_Fault": "Fault",      # YTA -> Fault
    "All_At_Fault": "Fault",       # ESH -> Fault (narrator shares blame)
    "Other_At_Fault": "No_Fault",  # NTA -> No Fault
    "No_One_At_Fault": "No_Fault", # NAH -> No Fault
    "Unclear": "Unclear",          # INFO -> Unclear
    "Error": "Error"
}

def standardize_judgment(raw_judgment: str) -> str:
    """
    Map format-specific judgments to standardized categories.

    Args:
        raw_judgment: The original LLM judgment (e.g., "YTA", "Main_At_Fault")

    Returns:
        Standardized judgment category
    """
    # Handle None or empty values
    if not raw_judgment:
        return "Unknown"

    # Strip whitespace and handle case variations
    judgment = raw_judgment.strip()

    # Direct mapping
    if judgment in JUDGMENT_MAP:
        return JUDGMENT_MAP[judgment]

    # Case-insensitive fallback
    for key, value in JUDGMENT_MAP.items():
        if judgment.lower() == key.lower():
            return value

    # If no mapping found, return as-is with warning
    logging.warning(f"No mapping found for judgment: '{judgment}'")
    return f"Unknown_{judgment}"


def standardize_binary_judgment(raw_judgment: str) -> str:
    """
    Map binary classification judgments to standardized categories.

    Args:
        raw_judgment: The original LLM judgment (e.g., "AT_FAULT", "NOT_AT_FAULT")

    Returns:
        Standardized binary judgment category ("Fault" or "No_Fault")
    """
    # Handle None or empty values
    if not raw_judgment:
        return "Unknown"

    # Strip whitespace and handle case variations
    judgment = raw_judgment.strip()

    # Direct mapping
    if judgment in BINARY_JUDGMENT_MAP:
        return BINARY_JUDGMENT_MAP[judgment]

    # Case-insensitive fallback
    for key, value in BINARY_JUDGMENT_MAP.items():
        if judgment.lower() == key.lower():
            return value

    # If no mapping found, return as-is with warning
    logging.warning(f"No binary mapping found for judgment: '{judgment}'")
    return f"Unknown_{judgment}"


def convert_multiclass_to_binary(standardized_judgment: str) -> str:
    """
    Convert a 4-class standardized judgment to binary (Fault/No_Fault).

    This enables comparison between binary and multi-class evaluations.

    Args:
        standardized_judgment: A standardized 4-class judgment
            (e.g., "Self_At_Fault", "Other_At_Fault")

    Returns:
        Binary classification ("Fault" or "No_Fault")
    """
    return MULTICLASS_TO_BINARY.get(standardized_judgment, "Unknown")


class EvaluationResponse(BaseModel):
    """Response schema for dilemma evaluations."""
    judgment: str
    explanation: str
    
    @classmethod
    def get_json_schema(cls) -> dict:
        """Generate JSON schema for structured outputs."""
        schema = {
            "type": "object",
            "properties": {
                "judgment": {
                    "type": "string",
                    "description": "The moral judgment for this dilemma"
                },
                "explanation": {
                    "type": "string", 
                    "description": "Explanation for the judgment"
                }
            },
            "required": ["judgment", "explanation"]
        }
        
        return schema