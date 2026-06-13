"""Demo script of game played with random agents."""

import argparse

import chess

from modules.agent import UIAgent
from modules.board import GUIBoard
from modules.game import Game

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo playing a game with an agent that picks random moves."
    )

    args = parser.parse_args()

    print("Start")

    board = GUIBoard()
    ui_agent = UIAgent(board)
    game = Game(ui_agent, ui_agent, board)

    game.play_game()

    print("End")

    print(
        f"Winner: {
            'White'
            if board.winner == chess.WHITE
            else 'BLACK'
            if board.winner == chess.BLACK
            else 'Draw'
        }"
    )
