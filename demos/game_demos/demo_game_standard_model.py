"""Script to demo use of a standard model in a game."""

import argparse
import time

import chess

from modules.agent import StandardAgent
from modules.board import ASCIIBoard, Board
from modules.game import Game
from modules.model import StandardModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo playing a game with the standard agent with model strain 0 gen 0."
    )
    parser.add_argument(
        "--gui", action="store_true", help="display the game with printoff in terminal"
    )

    args = parser.parse_args()

    print("Start")

    start = time.perf_counter()

    board = ASCIIBoard() if args.gui else Board()
    white_agent = StandardAgent(StandardModel(0, 0), confidence_factor=1.0)
    black_agent = StandardAgent(StandardModel(0, 0), confidence_factor=1.0)
    game = Game(white_agent, black_agent, board)

    game.play_game()

    end = time.perf_counter()

    print(f"Done with {((end - start) / board.half_move_count * 1e3):.3f} ms per move")
    print(
        f"Winner: {
            'White'
            if board.winner == chess.WHITE
            else 'BLACK'
            if board.winner == chess.BLACK
            else 'Draw'
        }"
    )
