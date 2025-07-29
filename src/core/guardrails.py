import re


# Patterns for out-of-scope or unwanted queries
OUT_OF_SCOPE_PATTERNS = [
    r"\bpython\b",
    r"\bcsv\b",
    r"\bcode\b",
    r"\binstall\b",
    r"\bpandas\b",
    r"\bscript\b",
    r"\bweather\b",
    r"\bcapital of\b",
    r"\bprogram\b",
    r"\bhow to\b",
    r"\bfile\b",
    r"\bpackage\b",
    r"\bimport\b",
    r"\bapi\b",
    r"\bcolumn\b",
    r"\bexcel\b",
    r"\bdocumentation\b",
    r"\bemail\b",
    r"\berror\b",
    r"\bbroken\b",
    r"\bsave\b",
    r"\bread\b",
    r"\bdataframe\b",
    r"\bdatabase\b",
    r"\bbug\b",
    r"\bcrash\b",
    r"\bmeta\b",
    r"\bdebug\b",
    r"\blist\b.*\bcolumns\b",
    r"\bopenai\b",
    r"\bgithub\b",
    r"\bpip\b",
    r"\bvenv\b"
]

# Patterns for direct or indirect medical advice requests
MEDICAL_ADVICE_PATTERNS = [
    r"\bshould I\b",
    r"\btake\b",
    r"\bprescribe\b",
    r"\bwhat medicine\b",
    r"\bhow do I treat\b",
    # r"\btreatment\b",
    # r"\bdose\b",
    r"\bcan I use\b",
    r"\bside effect\b",
    # r"\bprescription\b",
    # r"\bdiagnose\b",
    # r"\bdiagnosis\b",
    r"\bhelp me\b",
    r"\bneed advice\b",
    r"\bhow do I cure\b"
]

# Patterns for unwanted outputs in model answers (code, install, etc.)
UNWANTED_OUTPUT_PATTERNS = [
    r"```",             # Markdown code blocks
    r"\bimport\s+\w+",  # Python imports
    r"\bpip\s+install\b",
    r"\bpython\s+-m\b",
    r"\bdef\s+\w+\(",   # Python function definition
    r"\bclass\s+\w+\(",
    r"\b\.py\b",
    r"\bopen\([^)]+\)", # Python open file
    r"\.csv\b",
    r"\bfrom\s+\w+\s+import\s+\w+",
    r"#\s*Example",
    r"Install(ing)?\s+\w+",
    r"\bvenv\b",
    r"\brequirements\.txt\b",
    r"\bgit\s+clone\b",
    r"\b\.env\b",
    r"\brepl\b",
    r"^\s*\$",           # shell command prompt
    r"\bcommand line\b",
]

def is_out_of_scope(query: str) -> bool:
    for pat in OUT_OF_SCOPE_PATTERNS:
        if re.search(pat, query, re.IGNORECASE):
            return True
    return False

def is_medical_advice(query: str) -> bool:
    for pat in MEDICAL_ADVICE_PATTERNS:
        if re.search(pat, query, re.IGNORECASE):
            return True
    return False

def guardrail_response(query: str) -> str | None:
    if is_medical_advice(query):
        return (
            "I'm sorry, but I cannot provide medical advice or recommendations. "
            "Please consult a licensed healthcare professional for any medical concerns."
        )
    elif is_out_of_scope(query):
        return (
            "Sorry, I can only answer questions about the healthcare dataset, "
            "not about programming, code, or unrelated topics."
        )
    else:
        return None  # Query is in-scope

def output_guardrail(answer: str) -> str:
    """Final output filter: blocks code/installation/meta answers."""
    if not answer or not isinstance(answer, str):
        return answer
    for pat in UNWANTED_OUTPUT_PATTERNS:
        if re.search(pat, answer, re.IGNORECASE | re.MULTILINE):
            return (
                "Sorry, I can only answer questions about the healthcare dataset, "
                "not about programming, code, or unrelated topics."
            )
    return answer
