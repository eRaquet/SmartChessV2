"""Helper Functions for UI Elements."""

from dataclasses import astuple, fields
from pathlib import Path

import chess
from tabulate import tabulate

from smartchess.types import AgentLogEntry, GameLog, GameLogEntry, MoveLogEntry


def write_game(game_log: GameLog) -> None:
    """Write game to output (currently just a text file)."""
    game_headers = [f.name for f in fields(GameLogEntry)]
    agent_headers = ['agent_color', *[f.name for f in fields(AgentLogEntry)]]
    move_headers = [f.name for f in fields(MoveLogEntry)]

    game_data = [list(astuple(game_log.game))]
    agent_data = [[color, *astuple(game_log.agents[color])] for color in [chess.BLACK, chess.WHITE]]
    move_data = [list(astuple(move)) for move in game_log.moves]

    game_string = tabulate(game_data, headers=game_headers, tablefmt='grid')
    agent_string = tabulate(agent_data, headers=agent_headers, tablefmt='grid')
    move_string = tabulate(move_data, headers=move_headers, tablefmt='grid')

    path = Path('temp.txt')

    with path.open('w') as file:
        print('Game Data', file=file)
        print(game_string, file=file)
        print('\nAgent Data', file=file)
        print(agent_string, file=file)
        print('\nMove Data', file=file)
        print(move_string, file=file)
