"""Run a benchmark on the game play performance."""

import cProfile
import statistics
import subprocess

from smartchess.capabilities import require_capability
from smartchess.config import PROJECT_PATH
from smartchess.game import GameConfig, create_game
from smartchess.types import BoardStatus, GameLog
from smartchess.util import format_time

from .config import BenchmarkGameConfig


def benchmark_game(config: BenchmarkGameConfig) -> None:
    """

    Run a benchmark on a game.

    Parameters
    ----------
    config : BenchmarkGameConfig
        config for benchmarking
    """
    if config.profile:
        require_capability('profile')

        if config.profile_output is None:
            msg = 'No profile output filename specified.'
            raise ValueError(msg)

        profiler = cProfile.Profile()
        results = profiler.runcall(
            run_games, game_config=config.game, num_games=config.trials, num_warmups=config.warmup
        )

        path = PROJECT_PATH / 'artifacts' / 'profiles' / (config.profile_output + '.prof')
        path.parent.mkdir(exist_ok=True)
        profiler.dump_stats(path)

        print_results(results)

        subprocess.Popen(  # noqa: S603
            ['tuna', path],  # noqa: S607
        )
    else:
        results = run_games(config.game, config.trials, config.warmup)

        print_results(results)


def run_games(game_config: GameConfig, num_games: int, num_warmups: int) -> list[GameLog]:
    """

    Run a bunch of games and return each game log.

    Parameters
    ----------
    game_config : GameConfig
        config for game to play, should include already formed agents and models
    num_games : int
        number of games to run
    num_warmups : int
        number of warmups to run

    Returns
    -------
    list[GameLog]
        logs from run games, excluding warmups
    """
    if not game_config.collect:
        msg = 'Benchmarked games must be collected.'
        raise ValueError(msg)

    for _ in range(num_warmups):
        game = create_game(game_config)
        game.play_game()

    results = []
    for _ in range(num_games):
        game = create_game(game_config)
        log = game.play_game()
        if log is None:
            msg = 'No log available for this game.'
            raise ValueError(msg)
        results.append(log)

    return results


def print_results(results: list[GameLog]) -> None:
    """

    Print the results from a profile.

    Parameters
    ----------
    results : list[GameLog]
        list of game logs from game runs
    """
    num_games = len(results)
    game_times = [log.game.dt for log in results]
    total_time: int = sum(game_times)
    game_ply_count = [log.game.ply_number for log in results]
    move_times = [move.dt for log in results for move in log.moves]
    total_moves: int = len(move_times)
    num_white: int = sum(1 for log in results if log.game.result is BoardStatus.WHITE)
    num_black: int = sum(1 for log in results if log.game.result is BoardStatus.BLACK)
    num_draw: int = sum(1 for log in results if log.game.result is BoardStatus.DRAW)

    if sum(1 for log in results if log.game.result is BoardStatus.UNDECIDED) > 0:
        msg = 'Found undecided game in benchmark.'
        raise RuntimeError(msg)
    if sum(1 for log in results if log.game.result is BoardStatus.UNTERMINATED) > 0:
        msg = 'Found unterminated game in benchmark.'
        raise RuntimeError(msg)

    print(f'Benchmark of {num_games}:')
    print(f'    Total Moves            : {total_moves}')
    print(f'    Average Moves Per Game : {(total_moves / num_games):.1f} moves', end='')
    print(
        (
            f' (std: {statistics.stdev(game_ply_count):.1f} moves, '
            f'median: {statistics.median(game_ply_count)} moves, '
            f'max: {max(game_ply_count)} moves)'
        )
        if num_games > 1
        else ''
    )
    print(
        f'    Outcomes               : (White: {num_white}, Black: {num_black}, Draw: {num_draw})'
    )
    print(f'    Total Benchmark Time   : {format_time(total_time)}')
    print(f'    Average Time Per Game  : {format_time(total_time / num_games)}', end='')
    print(
        f' (std: {format_time(statistics.stdev(game_times))}, '
        f'median: {format_time(statistics.median(game_times))}, '
        f'max: {format_time(max(game_times))})'
        if num_games > 1
        else ''
    )
    print(
        f'    Average Time Per Ply   : {format_time(total_time / total_moves)}'
        f' (std: {format_time(statistics.stdev(move_times))}, '
        f'median: {format_time(statistics.median(move_times))}, '
        f'max: {format_time(max(move_times))})'
    )
