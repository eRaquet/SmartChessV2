"""Utils Module."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from .encoding import encode_board as encode_board
from .encoding import generate_observation as generate_observation
from .math import entropy as entropy
from .math import softmax as softmax

if TYPE_CHECKING:
    from collections.abc import Callable

    from smartchess.types import GameLog


def __getattr__(name: Literal['write_game']) -> Callable[[GameLog], None]:
    if name == 'write_game':
        try:
            from .ui import write_game
        except ModuleNotFoundError as error:
            if error.name == 'tabulate':
                msg = 'write_game requires `tabulate` package.  Install smartchess-v2[visual]'
                raise ImportError(msg) from error
            raise

        return write_game

    msg = f'module {__name__!r} has no attribute {name!r}'
    raise AttributeError(msg)
