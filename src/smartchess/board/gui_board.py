"""GUI Board."""

from typing import override

from smartchess.types import Action
from smartchess.ui import Display

from .standard_board import Board


class GUIBoard(Board):
    """Board with full pygame gui."""

    def __init__(self) -> None:
        super().__init__()

        self._display = Display()

    @override
    def _render(self) -> None:
        """Render board display."""
        self._display.display_board(self._board)

    def get_user_input(self) -> Action | None:
        """

        Check if the user has given GUI input, and return the move if possible.

        Returns
        -------
        Action | None
            Action selected by the user, or None if no action is yet selected
        """
        return self._display.get_user_input(self._board, self._moves)
