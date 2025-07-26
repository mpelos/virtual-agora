"""Security utilities for Virtual Agora.

This module provides functions for secure handling of sensitive data,
particularly API keys and other credentials.
"""

import hashlib
import hmac
import re
import secrets
from typing import Optional, Dict, Any, List


def mask_api_key(key: Optional[str], show_chars: int = 4) -> str:
    """Mask an API key for safe display.

    Args:
        key: API key to mask.
        show_chars: Number of characters to show at the end.

    Returns:
        Masked key showing only specified number of end characters.
    """
    if not key:
        return "<not set>"

    key = str(key).strip()
    if not key:
        return "<empty>"

    if len(key) <= show_chars * 2:
        # Too short to mask meaningfully
        return "*" * len(key)

    # Show first few and last few characters
    if show_chars > 0:
        return f"{key[:2]}{'*' * (len(key) - 2 - show_chars)}{key[-show_chars:]}"
    else:
        return "*" * len(key)


def mask_dict_values(data: Dict[str, Any], keys_to_mask: List[str]) -> Dict[str, Any]:
    """Mask sensitive values in a dictionary.

    Args:
        data: Dictionary containing potentially sensitive data.
        keys_to_mask: List of key names to mask.

    Returns:
        New dictionary with specified values masked.
    """
    masked_data = data.copy()

    for key in keys_to_mask:
        if key in masked_data and masked_data[key]:
            masked_data[key] = mask_api_key(str(masked_data[key]))

    return masked_data


def secure_compare(a: str, b: str) -> bool:
    """Compare two strings in constant time to prevent timing attacks.

    Args:
        a: First string.
        b: Second string.

    Returns:
        True if strings are equal, False otherwise.
    """
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


def validate_api_key_format(key: str, provider: str) -> tuple[bool, Optional[str]]:
    """Validate API key format for known providers.

    Args:
        key: API key to validate.
        provider: Provider name.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not key or not key.strip():
        return False, "API key is empty"

    # Known patterns (these may change over time)
    patterns = {
        "Google": {
            "pattern": r"^AIza[0-9A-Za-z\-_]{35}$",
            "description": "Google API keys typically start with 'AIza' followed by 35 characters",
        },
        "OpenAI": {
            "pattern": r"^sk-[A-Za-z0-9]{48}$",
            "description": "OpenAI API keys start with 'sk-' followed by 48 characters",
        },
        "Anthropic": {
            "pattern": r"^sk-ant-[A-Za-z0-9\-_]+$",
            "description": "Anthropic API keys start with 'sk-ant-' followed by alphanumeric characters",
        },
    }

    provider_info = patterns.get(provider)
    if not provider_info:
        # Unknown provider, can't validate format
        return True, None

    if re.match(provider_info["pattern"], key):
        return True, None
    else:
        return False, f"Invalid format. {provider_info['description']}"


def sanitize_for_logging(text: str, patterns: Optional[List[str]] = None) -> str:
    """Sanitize text for safe logging by removing potential secrets.

    Args:
        text: Text to sanitize.
        patterns: Additional regex patterns to match and remove.

    Returns:
        Sanitized text safe for logging.
    """
    # Default patterns for common secrets
    default_patterns = [
        r"AIza[0-9A-Za-z\-_]{35}",  # Google API keys
        r"sk-[A-Za-z0-9]{48}",  # OpenAI keys
        r"sk-ant-[A-Za-z0-9\-_]+",  # Anthropic keys
        r"[A-Za-z0-9+/]{40,}={0,2}",  # Base64 encoded strings (potential keys)
        r"[a-f0-9]{32,}",  # Hex strings (potential hashes/keys)
    ]

    all_patterns = default_patterns + (patterns or [])

    sanitized = text
    for pattern in all_patterns:
        sanitized = re.sub(
            pattern,
            lambda m: mask_api_key(m.group(), show_chars=4),
            sanitized,
            flags=re.IGNORECASE,
        )

    return sanitized


def generate_session_id() -> str:
    """Generate a secure random session ID.

    Returns:
        Hex string session ID.
    """
    return secrets.token_hex(16)


def hash_api_key(key: str) -> str:
    """Generate a hash of an API key for comparison without storing the key.

    Args:
        key: API key to hash.

    Returns:
        SHA-256 hash of the key.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def create_provider_help_message(provider: str) -> str:
    """Create a helpful message for obtaining API keys for a provider.

    Args:
        provider: Provider name.

    Returns:
        Help message with instructions.
    """
    help_messages = {
        "Google": (
            "To get a Google API key:\n"
            "1. Go to https://makersuite.google.com/app/apikey\n"
            "2. Click 'Create API Key'\n"
            "3. Copy the key and add it to your .env file as GOOGLE_API_KEY=<your-key>"
        ),
        "OpenAI": (
            "To get an OpenAI API key:\n"
            "1. Go to https://platform.openai.com/api-keys\n"
            "2. Click 'Create new secret key'\n"
            "3. Copy the key and add it to your .env file as OPENAI_API_KEY=<your-key>"
        ),
        "Anthropic": (
            "To get an Anthropic API key:\n"
            "1. Go to https://console.anthropic.com/account/keys\n"
            "2. Click 'Create Key'\n"
            "3. Copy the key and add it to your .env file as ANTHROPIC_API_KEY=<your-key>"
        ),
        "Grok": (
            "Grok API access is currently limited.\n"
            "Check the official Grok documentation for availability.\n"
            "Once you have a key, add it to your .env file as GROK_API_KEY=<your-key>"
        ),
    }

    return help_messages.get(
        provider,
        f"Please obtain an API key for {provider} and add it to your .env file.",
    )


def check_env_file_security(file_path: str) -> List[str]:
    """Check security of an environment file.

    Args:
        file_path: Path to the environment file.

    Returns:
        List of security warnings.
    """
    warnings = []

    try:
        import stat
        import os

        # Check file permissions
        file_stat = os.stat(file_path)
        mode = file_stat.st_mode

        # Check if world-readable
        if mode & stat.S_IROTH:
            warnings.append(f"File is world-readable. Run: chmod 600 {file_path}")

        # Check if world-writable
        if mode & stat.S_IWOTH:
            warnings.append(f"File is world-writable! Run: chmod 600 {file_path}")

        # Check ownership
        import pwd

        file_owner = pwd.getpwuid(file_stat.st_uid).pw_name
        current_user = pwd.getpwuid(os.getuid()).pw_name

        if file_owner != current_user:
            warnings.append(
                f"File is owned by '{file_owner}', not current user '{current_user}'"
            )

    except (ImportError, OSError, KeyError):
        # Can't check on this platform
        pass

    return warnings
