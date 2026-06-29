"""Script to demo use of a random model in a game."""

import argparse
import time

import chess

from modules.agent import StandardAgent
from modules.board import ASCIIBoard, Board, GUIBoard
from modules.collector import Collector
from modules.game import Game
from modules.model import RandomModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo playing a game with the standard agent with a random model."
    )
    parser.add_argument("--gui", action="store_true", help="display the game with GUI")
    parser.add_argument(
        "--ascii", action="store_true", help="display the game with ASCII in terminal"
    )
    parser.add_argument("--log", action="store_true", help="log game to database")

    args = parser.parse_args()

    print("Start")

    start = time.perf_counter()

    board = ASCIIBoard() if args.ascii else GUIBoard() if args.gui else Board()
    white_agent = StandardAgent(RandomModel(), confidence_factor=1.0)
    black_agent = StandardAgent(RandomModel(), confidence_factor=1.0)

    collector = Collector() if args.log else None
    game = Game(white_agent, black_agent, board, collector)

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
