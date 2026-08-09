# ruff: noqa: S101
"""Tests for display class."""

import os

from smartchess.board import BoardConfig, create_board

# make a blank rendering backend for testing
os.environ['SDL_VIDEODRIVER'] = 'dummy'

from smartchess.ui import GUIConfig, create_gui


def test_gui_render() -> None:
    """Test the render method of the GUI class."""
    gui_config = GUIConfig()
    gui = create_gui(gui_config)

    board_config = BoardConfig()
    board = create_board(board_config)

    gui.render(board.snapshot)

    board.step(0)

    gui.render(board.snapshot)

    gui.exit()


def test_gui_request_move() -> None:
    """Test the request_move method of the GUI class."""
    gui_config = GUIConfig()
    gui = create_gui(gui_config)

    board_config = BoardConfig()
    board = create_board(board_config)

    assert gui.request_move(board.snapshot) is None

    gui.exit()
