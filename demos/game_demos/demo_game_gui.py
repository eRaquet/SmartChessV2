"""Demo script of game played with random agents."""

import argparse

import chess

from modules.agent import RandomAgent, StandardAgent, UIAgent
from modules.board import GUIBoard
from modules.collector import Collector
from modules.game import LoggedGame, StandardGame
from modules.model import StandardModel
from modules.utils import write_game

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Demo playing a game with an agent that picks random moves."
    )
    parser.add_argument(
        "--white",
        nargs="*",
        default=["human"],
        help='Options: "h" or "human", "r" or "random", "<strain>", or "<strain> <generation>"',
    )
    parser.add_argument(
        "--black",
        nargs="*",
        default=["human"],
        help='Options: "h" or "human", "r" or "random", "<strain>", or "<strain> <generation>"',
    )
    parser.add_argument("--log", action="store_true", help="log game to database")

    args = parser.parse_args()

    board = GUIBoard()

    if args.white:
        if len(args.white) == 1:
            arg = args.white[0]
            if arg in ["h", "human"]:
                white_agent = UIAgent(board)
            elif arg in ["r", "random"]:
                white_agent = RandomAgent()
            elif arg.isdigit():
                white_agent = StandardAgent(StandardModel(int(arg)))
            else:
                msg = "Invalid argument for [--white] option"
                raise ValueError(msg)
        elif len(args.white) == 2:  # noqa: PLR2004
            strain = args.white[0]
            generation = args.white[1]
            if strain.isdigit() and generation.isdigit():
                white_agent = StandardAgent(StandardModel(int(strain), int(generation)))
            else:
                msg = "Invalid arguments for [--white] option"
                raise ValueError(msg)
        else:
            msg = "Invalid number of arguments for [--white] option"
            raise ValueError(msg)
    else:
        white_agent = UIAgent(board)

    if args.black:
        if len(args.black) == 1:
            arg = args.black[0]
            if arg in ["h", "human"]:
                black_agent = UIAgent(board)
            elif arg in ["r", "random"]:
                black_agent = RandomAgent()
            elif arg.isdigit():
                black_agent = StandardAgent(StandardModel(int(arg)))
            else:
                msg = "Invalid argument for [--black] option"
                raise ValueError(msg)
        elif len(args.black) == 2:  # noqa: PLR2004
            strain = args.black[0]
            generation = args.black[1]
            if strain.isdigit() and generation.isdigit():
                black_agent = StandardAgent(StandardModel(int(strain), int(generation)))
            else:
                msg = "Invalid arguments for [--black] option"
                raise ValueError(msg)
        else:
            msg = "Invalid number of arguments for [--black] option"
            raise ValueError(msg)
    else:
        black_agent = UIAgent(board)

    if args.log:
        collector = Collector()
        game = LoggedGame(white_agent, black_agent, board, collector)
    else:
        game = StandardGame(white_agent, black_agent, board)

    print("Start")

    log = game.play_game()

    print("End")

    if log is not None:
        write_game(log)

    print(
        f"Winner: {
            'White'
            if board.winner == chess.WHITE
            else 'BLACK'
            if board.winner == chess.BLACK
            else 'Draw'
        }"
    )
