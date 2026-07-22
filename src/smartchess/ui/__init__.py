"""UI Module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from .display import Display


def __getattr__(name: Literal['Display']) -> type[Display]:
    if name == 'Display':
        try:
            from .display import Display
        except ModuleNotFoundError as error:
            if error.name == 'pygame':
                msg = 'Display requires a graphics backend.  Install smartchess-v2[visual]'
                raise ImportError(msg) from error
            raise

        return Display

    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)
