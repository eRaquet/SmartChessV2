"""UI Agent."""

from typing import override

from smartchess.board import Board, GUIBoard
from smartchess.types import AgentDecision

from .agent_base import AgentBase


class UIAgent(AgentBase):
    """Agent that gets user input from a board with a GUI."""

    def __init__(self, board: GUIBoard) -> None:
        if type(board) is not GUIBoard:
            msg = 'UI Agents can only be instantiated from a GUI Board.'
            raise TypeError(msg)

        # core objects that a UIAgent contains
        self._board: GUIBoard = board

    @override
    def act(self, board: Board) -> AgentDecision:
        """

        Get the user input.

        Returns
        -------
        AgentDecision
            action to take, specified by user
        """
        action = None
        while action is None:
            action = self._board.get_user_input()

        return self._capture(action)
