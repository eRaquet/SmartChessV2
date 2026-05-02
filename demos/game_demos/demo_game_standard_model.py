"""Script to demo use of standard model in a game."""

import time

from modules.agent import StandardAgent
from modules.board import Board
from modules.chess_types import DisplayMode
from modules.game import Game
from modules.model import StandardModel

if __name__ == "__main__":
    print("Start")
    start = time.perf_counter()

    game = Game(
        StandardAgent(StandardModel(0, 0), confidence_factor=1.0),
        StandardAgent(StandardModel(0, 0), confidence_factor=1.0),
        Board(DisplayMode.GUI),
    )
    game.play_game()

    end = time.perf_counter()

    print(f"Done with {((end - start) / len(game.trajectory) * 1e3):.3f} ms per move")
