"""Demo script of game played with agents driven by random model."""

import time

from modules.agent import StandardAgent
from modules.board import Board
from modules.chess_types import DisplayMode
from modules.game import Game
from modules.model import RandomModel

if __name__ == "__main__":
    print("Start")
    start = time.perf_counter()

    game = Game(
        StandardAgent(RandomModel(), confidence_factor=1.0), StandardAgent(RandomModel(), confidence_factor=1.0), Board(DisplayMode.GUI)
    )
    game.play_game()

    end = time.perf_counter()

    print(f"Done with {((end - start) / len(game.trajectory) * 1e3):.3f} ms per move")
