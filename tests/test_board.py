# ruff: noqa: S101
"""File for testing tools module."""

from pathlib import Path

import numpy as np

from modules.board import Board

path = Path(__file__).parent


def test_board() -> None:
    """Integration test the Board class."""
