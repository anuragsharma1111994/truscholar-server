"""DateTime utilities for TruScholar application.

This module provides timezone-aware datetime utilities, formatting functions,
and time calculation helpers for consistent datetime handling.
"""

import time
from datetime import datetime, timedelta, timezone
from typing import Optional, Union

import pytz


def utc_now() -> datetime:
    """Get current UTC datetime.

    Returns:
        datetime: Current UTC datetime with timezone info
    """
    return datetime.now(timezone.utc)


def get_current_timestamp() -> int:
    """Get current Unix timestamp.

    Returns:
        int: Unix timestamp in seconds
    """
    return int(time.time())


def get_current_timestamp_ms() -> int:
    """Get current Unix timestamp in milliseconds.

    Returns:
        int: Unix timestamp in milliseconds
    """
    return int(time.time() * 1000)


def parse_datetime(
    dt_string: str,
    fmt: Optional[str] = None,
    timezone_aware: bool = True
) -> Optional[datetime]:
    """Parse datetime string to datetime object.

    Args:
        dt_string: DateTime string to parse
        fmt: Format string (if None, tries common formats)
        timezone_aware: Whether to make result timezone-aware

    Returns:
        datetime: Parsed datetime object or None if parsing fails
    """
    if not dt_string:
        return None

    # Common datetime formats to try
    formats = [
        fmt,
        "%Y-%m-%dT%H:%M:%S.%fZ",      # ISO format with microseconds
        "%Y-%m-%dT%H:%M:%SZ",         # ISO format without microseconds
        "%Y-%m-%dT%H:%M:%S.%f%z",     # ISO with timezone
        "%Y-%m-%dT%H:%M:%S%z",        # ISO with timezone, no microseconds
        "%Y-%m-%d %H:%M:%S",          # Standard format
        "%Y-%m-%d %H:%M:%S.%f",       # Standard with microseconds
        "%Y-%m-%d",                   # Date only
        "%d/%m/%Y %H:%M:%S",          # Indian format
        "%d/%m/%Y",                   # Indian date only
    ] if fmt is None else [fmt]

    for format_str in formats:
        if format_str is None:
            continue

        try:
            dt = datetime.strptime(dt_string, format_str)

            # Make timezone-aware if requested and not already
            if timezone_aware and dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            return dt

        except ValueError:
            continue

    return None


def format_datetime(
    dt: datetime,
    fmt: str = "%Y-%m-%d %H:%M:%S",
    timezone_name: Optional[str] = None
) -> str:
    """Format datetime object to string.

    Args:
        dt: DateTime object to format
        fmt: Format string
        timezone_name: Target timezone name (e.g., 'Asia/Kolkata')

    Returns:
        str: Formatted datetime string
    """
    if timezone_name:
        target_tz = pytz.timezone(timezone_name)
        if dt.tzinfo is None:
            # Assume UTC if no timezone info
            dt = dt.replace(tzinfo=timezone.utc)
        dt = dt.astimezone(target_tz)

    return dt.strftime(fmt)


def format_datetime_iso(dt: datetime) -> str:
    """Format datetime to ISO string.

    Args:
        dt: DateTime object to format

    Returns:
        str: ISO formatted datetime string
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def get_timezone_aware_datetime(
    dt: datetime,
    timezone_name: str = "UTC"
) -> datetime:
    """Convert naive datetime to timezone-aware datetime.

    Args:
        dt: DateTime object (naive or aware)
        timezone_name: Timezone name

    Returns:
        datetime: Timezone-aware datetime object
    """
    if dt.tzinfo is not None:
        # Already timezone-aware
        return dt

    tz = pytz.timezone(timezone_name)
    return tz.localize(dt)


def convert_timezone(
    dt: datetime,
    target_timezone: str
) -> datetime:
    """Convert datetime to different timezone.

    Args:
        dt: DateTime object
        target_timezone: Target timezone name

    Returns:
        datetime: Converted datetime object
    """
    if dt.tzinfo is None:
        # Assume UTC for naive datetime
        dt = dt.replace(tzinfo=timezone.utc)

    target_tz = pytz.timezone(target_timezone)
    return dt.astimezone(target_tz)


def get_indian_timezone_datetime(dt: Optional[datetime] = None) -> datetime:
    """Get datetime in Indian timezone (Asia/Kolkata).

    Args:
        dt: DateTime to convert (defaults to current UTC time)

    Returns:
        datetime: DateTime in Indian timezone
    """
    if dt is None:
        dt = utc_now()

    return convert_timezone(dt, "Asia/Kolkata")


def calculate_age(birth_date: datetime, reference_date: Optional[datetime] = None) -> int:
    """Calculate age from birth date.

    Args:
        birth_date: Birth date
        reference_date: Reference date (defaults to current date)

    Returns:
        int: Age in years
    """
    if reference_date is None:
        reference_date = utc_now()

    # Remove timezone info for calculation
    if birth_date.tzinfo is not None:
        birth_date = birth_date.replace(tzinfo=None)
    if reference_date.tzinfo is not None:
        reference_date = reference_date.replace(tzinfo=None)

    age = reference_date.year - birth_date.year

    # Adjust if birthday hasn't occurred this year
    if (reference_date.month, reference_date.day) < (birth_date.month, birth_date.day):
        age -= 1

    return age


def add_business_days(start_date: datetime, days: int) -> datetime:
    """Add business days to a date (excludes weekends).

    Args:
        start_date: Starting date
        days: Number of business days to add

    Returns:
        datetime: Resulting date
    """
    current_date = start_date
    days_added = 0

    while days_added < days:
        current_date += timedelta(days=1)
        # Monday = 0, Sunday = 6
        if current_date.weekday() < 5:  # Monday to Friday
            days_added += 1

    return current_date


def is_business_day(dt: datetime) -> bool:
    """Check if datetime falls on a business day.

    Args:
        dt: DateTime to check

    Returns:
        bool: True if business day (Monday-Friday)
    """
    return dt.weekday() < 5


def get_time_duration_string(start_time: datetime, end_time: datetime) -> str:
    """Get human-readable duration string.

    Args:
        start_time: Start datetime
        end_time: End datetime

    Returns:
        str: Duration string (e.g., "2 hours 30 minutes")
    """
    duration = end_time - start_time

    if duration.total_seconds() < 0:
        return "0 seconds"

    days = duration.days
    hours, remainder = divmod(duration.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    parts = []

    if days > 0:
        parts.append(f"{days} day{'s' if days != 1 else ''}")
    if hours > 0:
        parts.append(f"{hours} hour{'s' if hours != 1 else ''}")
    if minutes > 0:
        parts.append(f"{minutes} minute{'s' if minutes != 1 else ''}")
    if not parts and seconds > 0:
        parts.append(f"{seconds} second{'s' if seconds != 1 else ''}")

    if not parts:
        return "0 seconds"

    return " ".join(parts)


def get_relative_time_string(dt: datetime, reference_time: Optional[datetime] = None) -> str:
    """Get relative time string (e.g., '2 hours ago', 'in 3 days').

    Args:
        dt: Target datetime
        reference_time: Reference datetime (defaults to current time)

    Returns:
        str: Relative time string
    """
    if reference_time is None:
        reference_time = utc_now()

    # Ensure both datetimes are timezone-aware or naive
    if dt.tzinfo is None and reference_time.tzinfo is not None:
        dt = dt.replace(tzinfo=timezone.utc)
    elif dt.tzinfo is not None and reference_time.tzinfo is None:
        reference_time = reference_time.replace(tzinfo=timezone.utc)

    delta = dt - reference_time
    seconds = delta.total_seconds()

    if seconds == 0:
        return "now"

    future = seconds > 0
    seconds = abs(seconds)

    intervals = [
        (31536000, "year"),
        (2592000, "month"),
        (604800, "week"),
        (86400, "day"),
        (3600, "hour"),
        (60, "minute"),
        (1, "second"),
    ]

    for interval_seconds, name in intervals:
        if seconds >= interval_seconds:
            count = int(seconds // interval_seconds)
            plural_suffix = "s" if count != 1 else ""

            if future:
                return f"in {count} {name}{plural_suffix}"
            else:
                return f"{count} {name}{plural_suffix} ago"

    return "just now"


def is_datetime_expired(dt: datetime, expiry_hours: int = 24) -> bool:
    """Check if datetime has expired based on hours from now.

    Args:
        dt: DateTime to check
        expiry_hours: Hours until expiry

    Returns:
        bool: True if expired
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    expiry_time = dt + timedelta(hours=expiry_hours)
    return utc_now() > expiry_time


def get_start_of_day(dt: datetime) -> datetime:
    """Get start of day (00:00:00) for given datetime.

    Args:
        dt: DateTime object

    Returns:
        datetime: Start of day
    """
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)


def get_end_of_day(dt: datetime) -> datetime:
    """Get end of day (23:59:59.999999) for given datetime.

    Args:
        dt: DateTime object

    Returns:
        datetime: End of day
    """
    return dt.replace(hour=23, minute=59, second=59, microsecond=999999)


def get_start_of_week(dt: datetime) -> datetime:
    """Get start of week (Monday 00:00:00) for given datetime.

    Args:
        dt: DateTime object

    Returns:
        datetime: Start of week
    """
    days_since_monday = dt.weekday()
    start_of_week = dt - timedelta(days=days_since_monday)
    return get_start_of_day(start_of_week)


def get_end_of_week(dt: datetime) -> datetime:
    """Get end of week (Sunday 23:59:59.999999) for given datetime.

    Args:
        dt: DateTime object

    Returns:
        datetime: End of week
    """
    days_until_sunday = 6 - dt.weekday()
    end_of_week = dt + timedelta(days=days_until_sunday)
    return get_end_of_day(end_of_week)


def get_start_of_month(dt: datetime) -> datetime:
    """Get start of month (1st day 00:00:00) for given datetime.

    Args:
        dt: DateTime object

    Returns:
        datetime: Start of month
    """
    return dt.replace(day=1, hour=0, minute=0, second=0, microsecond=0)


def get_end_of_month(dt: datetime) -> datetime:
    """Get end of month (last day 23:59:59.999999) for given datetime.

    Args:
        dt: DateTime object

    Returns:
        datetime: End of month
    """
    if dt.month == 12:
        next_month = dt.replace(year=dt.year + 1, month=1, day=1)
    else:
        next_month = dt.replace(month=dt.month + 1, day=1)

    last_day_of_month = next_month - timedelta(days=1)
    return get_end_of_day(last_day_of_month)


def create_datetime_range(
    start_date: datetime,
    end_date: datetime,
    interval: timedelta = timedelta(days=1)
) -> list[datetime]:
    """Create a range of datetime objects.

    Args:
        start_date: Start datetime
        end_date: End datetime
        interval: Interval between datetimes

    Returns:
        list[datetime]: List of datetime objects
    """
    dates = []
    current_date = start_date

    while current_date <= end_date:
        dates.append(current_date)
        current_date += interval

    return dates


class DateTimeUtils:
    """Utility class for datetime operations with configuration."""

    def __init__(self, default_timezone: str = "UTC"):
        """Initialize with default timezone.

        Args:
            default_timezone: Default timezone name
        """
        self.default_timezone = default_timezone
        self._tz = pytz.timezone(default_timezone)

    def now(self) -> datetime:
        """Get current datetime in default timezone."""
        return datetime.now(self._tz)

    def to_default_timezone(self, dt: datetime) -> datetime:
        """Convert datetime to default timezone."""
        return convert_timezone(dt, self.default_timezone)

    def format_for_display(self, dt: datetime) -> str:
        """Format datetime for user display."""
        local_dt = self.to_default_timezone(dt)
        return format_datetime(local_dt, "%d %b %Y, %I:%M %p")

    def format_for_api(self, dt: datetime) -> str:
        """Format datetime for API response."""
        return format_datetime_iso(dt)


# Create a default instance for Indian timezone
indian_datetime_utils = DateTimeUtils("Asia/Kolkata")

# Export commonly used functions
__all__ = [
    "utc_now",
    "get_current_timestamp",
    "get_current_timestamp_ms",
    "parse_datetime",
    "format_datetime",
    "format_datetime_iso",
    "get_timezone_aware_datetime",
    "convert_timezone",
    "get_indian_timezone_datetime",
    "calculate_age",
    "add_business_days",
    "is_business_day",
    "get_time_duration_string",
    "get_relative_time_string",
    "is_datetime_expired",
    "get_start_of_day",
    "get_end_of_day",
    "get_start_of_week",
    "get_end_of_week",
    "get_start_of_month",
    "get_end_of_month",
    "create_datetime_range",
    "DateTimeUtils",
    "indian_datetime_utils",
]
