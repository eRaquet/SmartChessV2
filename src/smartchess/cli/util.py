"""Utils for CLI module."""

import re

import typer

from smartchess.config import DEFAULT_CONFIDENCE


# sentinel object for ommitted flags when None should be an argument
class Unset:
    """Sentinal class for unset parameters."""


UNSET = Unset()


FLOAT_OR_INF_PATTERN = r'([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|inf|Inf|INF)'
TUPLE_PATTERN = re.compile(
    r'^\s*\(\s*'
    f'{FLOAT_OR_INF_PATTERN}?'  # Optional First Value
    r'\s*,\s*'
    f'{FLOAT_OR_INF_PATTERN}?'  # Optional Second Value
    r'\s*\)\s*$'
)


def parse_confidence(value: str | Unset) -> float | None | tuple[float | None, float | None]:
    """Parse the confidence field."""
    if isinstance(value, Unset):
        return DEFAULT_CONFIDENCE
    match = TUPLE_PATTERN.match(value)
    if not match:
        return optional_float(value)
    return (optional_float(match.group(1)), optional_float(match.group(2)))


def optional_float(value: str) -> float | None:
    """Parse an optional float."""
    if value == 'inf':
        return None
    try:
        return float(value)
    except ValueError:
        msg = f'{value} must be a float'
        raise typer.BadParameter(msg) from None


def nonnegative_integer(value: str, label: str) -> int:
    """Parse a nonnegative integer or produce a CLI validation error."""
    try:
        parsed = int(value)
    except ValueError:
        msg = f'{label} must be an integer, received {value!r}.'
        raise typer.BadParameter(msg) from None

    if parsed < 0:
        msg = f'{label} must be nonnegative, received {parsed}.'
        raise typer.BadParameter(msg)

    return parsed
