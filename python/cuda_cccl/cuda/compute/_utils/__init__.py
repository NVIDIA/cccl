import re

__all__ = ["sanitize_identifier"]


def sanitize_identifier(name: str) -> str:
    """Sanitize a name to be a valid Python/LLVM identifier.

    This replaces any character that isn't alphanumeric or underscore with
    an underscore. This is needed because:
    - Lambda functions have __name__ = "<lambda>" which contains angle brackets
    - Python identifiers and LLVM/NVVM global names don't allow special characters

    Args:
        name: The name to sanitize (e.g., function __name__)

    Returns:
        A sanitized name safe for use as a Python identifier or LLVM symbol
    """
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)
