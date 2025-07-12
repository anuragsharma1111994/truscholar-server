"""Helper utility functions for TruScholar application.

This module provides common utility functions for string manipulation,
data processing, formatting, and other general-purpose operations.
"""

import re
import secrets
import string
import uuid
from typing import Any, Dict, List, Optional, Union
from urllib.parse import quote, unquote


def generate_random_string(
    length: int = 8,
    include_uppercase: bool = True,
    include_lowercase: bool = True,
    include_digits: bool = True,
    include_special: bool = False,
    exclude_ambiguous: bool = True
) -> str:
    """Generate a random string with specified characteristics.

    Args:
        length: Length of the string
        include_uppercase: Include uppercase letters
        include_lowercase: Include lowercase letters
        include_digits: Include digits
        include_special: Include special characters
        exclude_ambiguous: Exclude ambiguous characters (0, O, I, l, etc.)

    Returns:
        str: Random string
    """
    characters = ""

    if include_lowercase:
        chars = string.ascii_lowercase
        if exclude_ambiguous:
            chars = chars.replace('l', '').replace('o', '')
        characters += chars

    if include_uppercase:
        chars = string.ascii_uppercase
        if exclude_ambiguous:
            chars = chars.replace('I', '').replace('O', '')
        characters += chars

    if include_digits:
        chars = string.digits
        if exclude_ambiguous:
            chars = chars.replace('0', '').replace('1', '')
        characters += chars

    if include_special:
        chars = "!@#$%^&*"
        characters += chars

    if not characters:
        raise ValueError("At least one character type must be included")

    return ''.join(secrets.choice(characters) for _ in range(length))


def generate_uuid() -> str:
    """Generate a UUID4 string.

    Returns:
        str: UUID4 string
    """
    return str(uuid.uuid4())


def generate_access_code(length: int = 8) -> str:
    """Generate an access code for reports or sessions.

    Args:
        length: Length of the access code

    Returns:
        str: Access code
    """
    return generate_random_string(
        length=length,
        include_uppercase=True,
        include_lowercase=False,
        include_digits=True,
        include_special=False,
        exclude_ambiguous=True
    )


def mask_sensitive_data(
    data: str,
    visible_chars: int = 4,
    mask_char: str = "*"
) -> str:
    """Mask sensitive data leaving only last few characters visible.

    Args:
        data: Data to mask
        visible_chars: Number of characters to leave visible
        mask_char: Character to use for masking

    Returns:
        str: Masked data
    """
    if not data or len(data) <= visible_chars:
        return mask_char * len(data) if data else ""

    masked_length = len(data) - visible_chars
    return mask_char * masked_length + data[-visible_chars:]


def mask_phone_number(phone: str) -> str:
    """Mask phone number for privacy.

    Args:
        phone: Phone number to mask

    Returns:
        str: Masked phone number
    """
    return mask_sensitive_data(phone, visible_chars=4)


def mask_email(email: str) -> str:
    """Mask email address for privacy.

    Args:
        email: Email to mask

    Returns:
        str: Masked email
    """
    if '@' not in email:
        return mask_sensitive_data(email, visible_chars=2)

    local, domain = email.split('@', 1)
    masked_local = mask_sensitive_data(local, visible_chars=2)
    return f"{masked_local}@{domain}"


def sanitize_string(
    text: str,
    allowed_chars: Optional[str] = None,
    replace_char: str = "_"
) -> str:
    """Sanitize string by removing or replacing unwanted characters.

    Args:
        text: Text to sanitize
        allowed_chars: Regex pattern of allowed characters
        replace_char: Character to replace unwanted characters with

    Returns:
        str: Sanitized string
    """
    if not text:
        return ""

    if allowed_chars is None:
        # Default: alphanumeric, spaces, hyphens, underscores
        allowed_chars = r'[^a-zA-Z0-9\s\-_]'

    # Replace unwanted characters
    sanitized = re.sub(allowed_chars, replace_char, text)

    # Remove multiple consecutive replace_chars
    if replace_char != ' ':
        sanitized = re.sub(f'{re.escape(replace_char)}+', replace_char, sanitized)

    # Strip leading/trailing replace_chars
    return sanitized.strip(replace_char + ' ')


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file storage.

    Args:
        filename: Filename to sanitize

    Returns:
        str: Sanitized filename
    """
    if not filename:
        return "file"

    # Remove or replace dangerous characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)

    # Remove control characters
    sanitized = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', sanitized)

    # Limit length
    if len(sanitized) > 255:
        name, ext = sanitized.rsplit('.', 1) if '.' in sanitized else (sanitized, '')
        max_name_length = 255 - len(ext) - 1 if ext else 255
        sanitized = name[:max_name_length] + ('.' + ext if ext else '')

    # Ensure it's not empty and doesn't start with dot
    if not sanitized or sanitized.startswith('.'):
        sanitized = 'file' + (sanitized if sanitized.startswith('.') else '')

    return sanitized


def truncate_string(text: str, max_length: int, suffix: str = "...") -> str:
    """Truncate string to maximum length with optional suffix.

    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated

    Returns:
        str: Truncated string
    """
    if not text or len(text) <= max_length:
        return text

    if len(suffix) >= max_length:
        return text[:max_length]

    return text[:max_length - len(suffix)] + suffix


def slugify(text: str, max_length: int = 50) -> str:
    """Convert text to URL-friendly slug.

    Args:
        text: Text to slugify
        max_length: Maximum length of slug

    Returns:
        str: URL-friendly slug
    """
    if not text:
        return ""

    # Convert to lowercase and replace spaces with hyphens
    slug = text.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)

    # Remove leading/trailing hyphens
    slug = slug.strip('-')

    # Truncate if necessary
    if len(slug) > max_length:
        slug = slug[:max_length].rstrip('-')

    return slug or "item"


def calculate_percentage(part: Union[int, float], total: Union[int, float]) -> float:
    """Calculate percentage with zero division protection.

    Args:
        part: Part value
        total: Total value

    Returns:
        float: Percentage (0-100)
    """
    if total == 0:
        return 0.0
    return (part / total) * 100


def round_to_decimal_places(value: float, decimal_places: int = 2) -> float:
    """Round value to specified decimal places.

    Args:
        value: Value to round
        decimal_places: Number of decimal places

    Returns:
        float: Rounded value
    """
    return round(value, decimal_places)


def format_number(
    number: Union[int, float],
    decimal_places: int = 2,
    use_commas: bool = True
) -> str:
    """Format number for display.

    Args:
        number: Number to format
        decimal_places: Number of decimal places
        use_commas: Whether to use comma separators

    Returns:
        str: Formatted number string
    """
    if isinstance(number, int) and decimal_places == 0:
        formatted = str(number)
    else:
        formatted = f"{number:.{decimal_places}f}"

    if use_commas:
        # Add comma separators
        parts = formatted.split('.')
        parts[0] = '{:,}'.format(int(parts[0]))
        formatted = '.'.join(parts)

    return formatted


def deep_merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries.

    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)

    Returns:
        Dict[str, Any]: Merged dictionary
    """
    result = dict1.copy()

    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def flatten_dict(
    data: Dict[str, Any],
    parent_key: str = "",
    separator: str = "."
) -> Dict[str, Any]:
    """Flatten nested dictionary.

    Args:
        data: Dictionary to flatten
        parent_key: Parent key prefix
        separator: Key separator

    Returns:
        Dict[str, Any]: Flattened dictionary
    """
    items = []

    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else key

        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, separator).items())
        else:
            items.append((new_key, value))

    return dict(items)


def unflatten_dict(
    data: Dict[str, Any],
    separator: str = "."
) -> Dict[str, Any]:
    """Unflatten dictionary with dot notation keys.

    Args:
        data: Flattened dictionary
        separator: Key separator

    Returns:
        Dict[str, Any]: Nested dictionary
    """
    result = {}

    for key, value in data.items():
        keys = key.split(separator)
        current = result

        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value

    return result


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks of specified size.

    Args:
        lst: List to chunk
        chunk_size: Size of each chunk

    Returns:
        List[List[Any]]: List of chunks
    """
    if chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def remove_duplicates(lst: List[Any], preserve_order: bool = True) -> List[Any]:
    """Remove duplicates from list.

    Args:
        lst: List with potential duplicates
        preserve_order: Whether to preserve original order

    Returns:
        List[Any]: List without duplicates
    """
    if preserve_order:
        seen = set()
        result = []
        for item in lst:
            if item not in seen:
                seen.add(item)
                result.append(item)
        return result
    else:
        return list(set(lst))


def convert_to_snake_case(text: str) -> str:
    """Convert text to snake_case.

    Args:
        text: Text to convert

    Returns:
        str: Snake case text
    """
    # Handle camelCase and PascalCase
    text = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)

    # Handle spaces and other separators
    text = re.sub(r'[\s\-\.]+', '_', text)

    # Convert to lowercase and clean up
    return re.sub(r'_+', '_', text.lower()).strip('_')


def convert_to_camel_case(text: str, pascal_case: bool = False) -> str:
    """Convert text to camelCase or PascalCase.

    Args:
        text: Text to convert
        pascal_case: Whether to use PascalCase (first letter uppercase)

    Returns:
        str: Camel case text
    """
    # Split on common separators
    words = re.split(r'[\s\-_\.]+', text.lower())

    # Filter out empty words
    words = [word for word in words if word]

    if not words:
        return ""

    # Convert to camelCase
    if pascal_case:
        return ''.join(word.capitalize() for word in words)
    else:
        return words[0] + ''.join(word.capitalize() for word in words[1:])


def extract_numbers(text: str) -> List[float]:
    """Extract all numbers from text.

    Args:
        text: Text to extract numbers from

    Returns:
        List[float]: List of extracted numbers
    """
    pattern = r'-?\d+\.?\d*'
    matches = re.findall(pattern, text)
    return [float(match) for match in matches if match]


def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text.

    Args:
        text: Text to extract emails from

    Returns:
        List[str]: List of email addresses
    """
    pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(pattern, text)


def extract_phone_numbers(text: str) -> List[str]:
    """Extract Indian phone numbers from text.

    Args:
        text: Text to extract phone numbers from

    Returns:
        List[str]: List of phone numbers
    """
    # Pattern for Indian mobile numbers
    patterns = [
        r'\b[6-9]\d{9}\b',  # 10-digit mobile
        r'\+91[6-9]\d{9}\b',  # With country code
        r'\b91[6-9]\d{9}\b',  # With country code without +
    ]

    numbers = []
    for pattern in patterns:
        numbers.extend(re.findall(pattern, text))

    # Clean and normalize
    cleaned_numbers = []
    for number in numbers:
        # Remove country code
        if number.startswith('+91'):
            number = number[3:]
        elif number.startswith('91') and len(number) == 12:
            number = number[2:]

        if len(number) == 10 and number[0] in '6789':
            cleaned_numbers.append(number)

    return remove_duplicates(cleaned_numbers)


def url_encode(text: str) -> str:
    """URL encode text.

    Args:
        text: Text to encode

    Returns:
        str: URL encoded text
    """
    return quote(text, safe='')


def url_decode(text: str) -> str:
    """URL decode text.

    Args:
        text: Text to decode

    Returns:
        str: URL decoded text
    """
    return unquote(text)


def is_valid_json(text: str) -> bool:
    """Check if text is valid JSON.

    Args:
        text: Text to check

    Returns:
        bool: True if valid JSON
    """
    try:
        import json
        json.loads(text)
        return True
    except (ValueError, TypeError):
        return False


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely parse JSON with default fallback.

    Args:
        text: JSON text to parse
        default: Default value if parsing fails

    Returns:
        Any: Parsed JSON or default value
    """
    try:
        import json
        return json.loads(text)
    except (ValueError, TypeError):
        return default


def get_file_extension(filename: str) -> str:
    """Get file extension from filename.

    Args:
        filename: Filename

    Returns:
        str: File extension (without dot)
    """
    if '.' not in filename:
        return ""
    return filename.rsplit('.', 1)[-1].lower()


def get_file_size_human_readable(size_bytes: int) -> str:
    """Convert file size to human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        str: Human readable size
    """
    if size_bytes == 0:
        return "0 B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0

    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024
        i += 1

    return f"{size_bytes:.1f} {size_names[i]}"


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace in text.

    Args:
        text: Text to normalize

    Returns:
        str: Text with normalized whitespace
    """
    if not text:
        return ""

    # Replace multiple whitespace with single space
    normalized = re.sub(r'\s+', ' ', text.strip())
    return normalized


def clean_text(text: str) -> str:
    """Clean text by removing extra whitespace and normalizing.

    Args:
        text: Text to clean

    Returns:
        str: Cleaned text
    """
    if not text:
        return ""

    # Remove leading/trailing whitespace
    cleaned = text.strip()

    # Normalize whitespace
    cleaned = normalize_whitespace(cleaned)

    # Remove zero-width characters
    cleaned = re.sub(r'[\u200b-\u200d\ufeff]', '', cleaned)

    return cleaned


# Export all helper functions
__all__ = [
    "generate_random_string",
    "generate_uuid",
    "generate_access_code",
    "mask_sensitive_data",
    "mask_phone_number",
    "mask_email",
    "sanitize_string",
    "sanitize_filename",
    "truncate_string",
    "slugify",
    "calculate_percentage",
    "round_to_decimal_places",
    "format_number",
    "deep_merge_dicts",
    "flatten_dict",
    "unflatten_dict",
    "chunk_list",
    "remove_duplicates",
    "convert_to_snake_case",
    "convert_to_camel_case",
    "extract_numbers",
    "extract_emails",
    "extract_phone_numbers",
    "url_encode",
    "url_decode",
    "is_valid_json",
    "safe_json_loads",
    "get_file_extension",
    "get_file_size_human_readable",
    "normalize_whitespace",
    "clean_text",
]
