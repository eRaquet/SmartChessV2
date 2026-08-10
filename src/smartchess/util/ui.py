"""Helper Functions for UI Elements."""

from dataclasses import astuple, fields
from pathlib import Path

import chess

from smartchess.capabilities import require_capability
from smartchess.types import AgentLogEntry, GameLog, GameLogEntry, MoveLogEntry


def write_game(game_log: GameLog, path: Path) -> None:
    """Write game to output (currently just a text file)."""
    require_capability('table')

    from tabulate import tabulate  # noqa: PLC0415

    game_headers = [f.name for f in fields(GameLogEntry)]
    agent_headers = ['agent_color', *[f.name for f in fields(AgentLogEntry)]]
    move_headers = [f.name for f in fields(MoveLogEntry)]

    game_data = [list(astuple(game_log.game))]
    agent_data = [[color, *astuple(game_log.agents[color])] for color in [chess.BLACK, chess.WHITE]]
    move_data = [list(astuple(move)) for move in game_log.moves]

    game_string = tabulate(game_data, headers=game_headers, tablefmt='grid')
    agent_string = tabulate(agent_data, headers=agent_headers, tablefmt='grid')
    move_string = tabulate(move_data, headers=move_headers, tablefmt='grid')

    # make parent directories, if not already present
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open('w') as file:
        print('Game Data', file=file)
        print(game_string, file=file)
        print('\nAgent Data', file=file)
        print(agent_string, file=file)
        print('\nMove Data', file=file)
        print(move_string, file=file)


# set the maximum quantity before the time will be converted into the next unit up
_UNIT_THRESHOLD = 100
_UNIT_PREFIX = ['n', 'µ', 'm', '', 'K', 'M', 'G', 'T']


def format_time(time: float, decimals: int = 3) -> str:
    """

    Turn a time in nanoseconds into a formatting time string based on it's size.

    Parameters
    ----------
    time : float
        time (or time interval) in nanoseconds
    decimals : int
        number of decimal places to use, default 3

    Returns
    -------
    str
        formatting time string
    """
    unit = 0

    while time > _UNIT_THRESHOLD:
        time /= 1e3
        unit += 1

    return f'{time:.{decimals}f} {_UNIT_PREFIX[unit]}s'
