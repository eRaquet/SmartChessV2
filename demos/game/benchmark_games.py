"""Benchmark full game playback."""

from __future__ import annotations

import argparse
import cProfile
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TextIO

import chess

from smartchess.agent import ModelAgent, RandomAgent
from smartchess.board import Board
from smartchess.config import PROJECT_PATH
from smartchess.game import LoggedGame, StandardGame
from smartchess.model import InferenceModel
from smartchess.pipeline import Collector
from smartchess.types import BoardStatus, Outcome

NANOSECONDS_PER_SECOND = 1_000_000_000
MILLISECONDS_PER_SECOND = 1_000
DEFAULT_GAMES = 100
DEFAULT_PROFILING_OUTPUT = PROJECT_PATH / 'artifacts' / 'profiles' / 'run.prof'


@dataclass(frozen=True, slots=True)
class GameResult:
    """Timing and outcome data for one benchmark game."""

    plies: int
    elapsed_seconds: float
    outcome: Outcome


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregate timing data for a benchmark run."""

    games: int
    warmup_games: int
    elapsed_seconds: float
    results: list[GameResult]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Benchmark SmartChessV2 by running complete games.',
        epilog=(
            'Examples:\n'
            '  uv run python benchmarks/run_games.py --games 200\n'
            '  uv run python benchmarks/run_games.py --games 50 --agent inference\n'
            '  uv run python benchmarks/run_games.py --games 20 --profile\n'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--games', type=int, default=DEFAULT_GAMES, help='timed games to run')
    parser.add_argument(
        '--warmup-games',
        type=int,
        default=1,
        help='untimed games to run before collecting benchmark timings',
    )
    parser.add_argument(
        '--agent',
        choices=('random', 'inference'),
        default='random',
        help=(
            'agent pair to benchmark: random avoids model inference; inference loads '
            'InferenceModel of strains 0 and 1'
        ),
    )
    parser.add_argument(
        '--profile',
        action='store_true',
        help='optional profiler to run around the benchmark body',
    )
    parser.add_argument(
        '--profile-output',
        type=Path,
        default=DEFAULT_PROFILING_OUTPUT,
        help='write cProfile stats to this path when --profile is used',
    )
    parser.add_argument('--log', action='store_true', help='run logging during games')
    return parser.parse_args()


def build_agents(
    agent_kind: Literal['random', 'inference'],
) -> tuple[RandomAgent | ModelAgent, RandomAgent | ModelAgent]:
    """Create the white and black agents for one game."""
    if agent_kind == 'random':
        return RandomAgent(), RandomAgent()
    return ModelAgent(InferenceModel(0)), ModelAgent(InferenceModel(1))


def play_one_game(
    agents: dict[chess.Color, RandomAgent | ModelAgent],
    *,
    log: bool,
) -> GameResult:
    """Play and time one benchmark game."""
    start_ns = time.perf_counter_ns()

    board = Board()

    if log:
        collector = Collector()
        game = LoggedGame(agents[chess.WHITE], agents[chess.BLACK], board, collector)

    else:
        game = StandardGame(agents[chess.WHITE], agents[chess.BLACK], board)

    game.play_game()

    elapsed_seconds = (time.perf_counter_ns() - start_ns) / NANOSECONDS_PER_SECOND
    return GameResult(
        plies=board.half_move_count,
        elapsed_seconds=elapsed_seconds,
        outcome=board.outcome,
    )


def run_games(
    *,
    games: int,
    warmup_games: int,
    agent_kind: Literal['random', 'inference'],
    log: bool,
) -> BenchmarkResult:
    """Run warmup and timed benchmark games."""
    white_agent, black_agent = build_agents(agent_kind)
    agents = {chess.WHITE: white_agent, chess.BLACK: black_agent}

    return collect_benchmark(
        agents=agents,
        games=games,
        warmup_games=warmup_games,
        log=log,
    )


def collect_benchmark(
    *,
    agents: dict[chess.Color, RandomAgent | ModelAgent],
    games: int,
    warmup_games: int,
    log: bool,
) -> BenchmarkResult:
    """Run warmup and timed games with already constructed agents."""
    for _ in range(warmup_games):
        play_one_game(
            agents,
            log=log,
        )

    benchmark_start_ns = time.perf_counter_ns()
    results = [
        play_one_game(
            agents,
            log=log,
        )
        for _ in range(games)
    ]
    elapsed_seconds = (time.perf_counter_ns() - benchmark_start_ns) / NANOSECONDS_PER_SECOND

    return BenchmarkResult(
        games=games,
        warmup_games=warmup_games,
        elapsed_seconds=elapsed_seconds,
        results=results,
    )


def format_rate(seconds: float, count: int) -> str:
    """Format a seconds-per-unit rate in milliseconds."""
    if count == 0:
        return 'n/a'
    return f'{seconds / count * MILLISECONDS_PER_SECOND:.3f} ms'


def print_summary(result: BenchmarkResult, *, output: TextIO = sys.stdout) -> None:
    """Print benchmark summary statistics."""
    plies = [game.plies for game in result.results]
    game_seconds = [game.elapsed_seconds for game in result.results]
    total_plies = sum(plies)
    outcomes = {
        status.name: sum(game.outcome.status == status for game in result.results)
        for status in (
            BoardStatus.WHITE,
            BoardStatus.BLACK,
            BoardStatus.DRAW,
            BoardStatus.UNDECIDED,
        )
    }

    print('Benchmark complete', file=output)
    print(f'  games:        {result.games}', file=output)
    print(f'  warmups:      {result.warmup_games}', file=output)
    print(f'  total plies:  {total_plies}', file=output)
    print(f'  outcomes:     {outcomes}', file=output)
    print(f'  total time:   {result.elapsed_seconds:.3f} s', file=output)
    print(f'  per game:     {format_rate(result.elapsed_seconds, result.games)}', file=output)
    print(f'  per ply:      {format_rate(result.elapsed_seconds, total_plies)}', file=output)

    if result.results:
        print(
            f'  game plies:   median={statistics.median(plies):.1f}, max={max(plies)}',
            file=output,
        )
        print(
            '  game time:    '
            f'median={statistics.median(game_seconds) * MILLISECONDS_PER_SECOND:.3f} ms, '
            f'max={max(game_seconds) * MILLISECONDS_PER_SECOND:.3f} ms',
            file=output,
        )


def run_profiled(args: argparse.Namespace) -> BenchmarkResult:
    """Run the benchmark under cProfile."""
    profiler = cProfile.Profile()
    result = profiler.runcall(
        run_games,
        games=args.games,
        warmup_games=args.warmup_games,
        agent_kind=args.agent,
        log=args.log,
    )

    if args.profile_output is not None:
        args.profile_output.parent.mkdir(parents=True, exist_ok=True)
        profiler.dump_stats(args.profile_output)

    return result


def main() -> None:
    """Run the benchmark CLI."""
    args = parse_args()

    if args.profile:
        result = run_profiled(args)
    else:
        result = run_games(
            games=args.games,
            warmup_games=args.warmup_games,
            agent_kind=args.agent,
            log=args.log,
        )

    print_summary(result)


if __name__ == '__main__':
    main()
