"""Demo script of game played with random agents."""

import time

from modules.agent import RandomAgent
from modules.board import Board
from modules.chess_types import DisplayMode
from modules.game import Game

if __name__ == "__main__":
    print("Start")
    start = time.perf_counter()

    game = Game(RandomAgent(), RandomAgent(), Board(DisplayMode.GUI))
    game.play_game()

    end = time.perf_counter()

    print(f"Done with {((end - start) / len(game.trajectory) * 1e3):.3f} ms per move")
