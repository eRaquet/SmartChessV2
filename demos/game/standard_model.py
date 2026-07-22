"""Script to demo use of a standard model in a game."""

import argparse
import time

import chess

from smartchess.agent import ModelAgent
from smartchess.board import Board, GUIBoard
from smartchess.game import LoggedGame, StandardGame
from smartchess.model import InferenceModel
from smartchess.pipeline import Collector
from smartchess.util import write_game

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Demo playing a game with the standard agent with model strain 0 gen 0.',
    )
    parser.add_argument('--gui', action='store_true', help='display the game with GUI')
    parser.add_argument('--log', action='store_true', help='log game to database')

    args = parser.parse_args()

    print('Start')

    start = time.perf_counter()

    board = GUIBoard() if args.gui else Board()
    white_agent = ModelAgent(InferenceModel(0, 0), confidence_factor=1.0)
    black_agent = ModelAgent(InferenceModel(0, 0), confidence_factor=1.0)

    if args.log:
        collector = Collector()
        game = LoggedGame(white_agent, black_agent, board, collector)
    else:
        game = StandardGame(white_agent, black_agent, board)

    log = game.play_game()

    end = time.perf_counter()

    if log is not None:
        write_game(log)

    print(f'Done with {((end - start) / board.half_move_count * 1e3):.3f} ms per move')
    print(
        f'Winner: {
            "White"
            if board.winner == chess.WHITE
            else "BLACK"
            if board.winner == chess.BLACK
            else "Draw"
        }',
    )
