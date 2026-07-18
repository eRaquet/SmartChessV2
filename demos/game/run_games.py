"""Benchmark full game playback."""

from __future__ import annotations

import argparse
import cProfile
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import chess
import numpy as np

from modules.agent import RandomAgent, StandardAgent
from modules.board import Board
from modules.chess_types import BoardStatus, Outcome
from modules.collector import Collector
from modules.game import LoggedGame, StandardGame
from modules.model import RandomModel, StandardModel

NANOSECONDS_PER_SECOND = 1_000_000_000
MILLISECONDS_PER_SECOND = 1_000
DEFAULT_GAMES = 100
DEFAULT_PROFILING_OUTPUT = Path("run.prof")


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
        description="Benchmark SmartChessV2 by running complete games.",
        epilog=(
            "Examples:\n"
            "  uv run python benchmarks/run_games.py --games 200\n"
            "  uv run python benchmarks/run_games.py --games 50 --agent random-model\n"
            "  uv run python benchmarks/run_games.py --games 20 --profile cprofile\n"
            "  uv run kernprof -l -v benchmarks/run_games.py --games 20"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="timed games to run")
    parser.add_argument(
        "--warmup-games",
        type=int,
        default=0,
        help="untimed games to run before collecting benchmark timings",
    )
    parser.add_argument(
        "--agent",
        choices=("random", "random-model", "standard-model"),
        default="random",
        help=(
            "agent pair to benchmark: random avoids model inference; random-model uses "
            "StandardAgent with RandomModel; standard-model loads StandardModel"
        ),
    )
    parser.add_argument("--strain", type=int, default=0, help="StandardModel strain")
    parser.add_argument(
        "--generation", type=int, default=0, help="StandardModel generation for standard-model"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=1.0,
        help="StandardAgent confidence factor for model-backed agents",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="seed NumPy's global random state for repeatable surrounding code",
    )
    parser.add_argument(
        "--profile",
        choices=("none", "cprofile"),
        default="none",
        help="optional profiler to run around the benchmark body",
    )
    parser.add_argument(
        "--profile-output",
        type=Path,
        default=DEFAULT_PROFILING_OUTPUT,
        help="write cProfile stats to this path when --profile cprofile is used",
    )
    parser.add_argument("--log", action="store_true", help="run logging during games")
    return parser.parse_args()


def build_agents(
    agent_kind: str,
    *,
    strain: int,
    generation: int,
    confidence: float,
) -> tuple[RandomAgent | StandardAgent, RandomAgent | StandardAgent]:
    """Create the white and black agents for one game."""
    if agent_kind == "random":
        return RandomAgent(), RandomAgent()

    if agent_kind == "random-model":
        return (
            StandardAgent(RandomModel(), confidence_factor=confidence),
            StandardAgent(RandomModel(), confidence_factor=confidence),
        )

    model = StandardModel(strain, generation)
    return (
        StandardAgent(model, confidence_factor=confidence),
        StandardAgent(model, confidence_factor=confidence),
    )


def seed_random_generators(seed: int) -> None:
    """Seed random generators used by benchmark agent choices."""
    RandomAgent._rng = np.random.default_rng(seed)  # noqa: SLF001
    StandardAgent._rng = np.random.default_rng(seed + 1)  # noqa: SLF001
    RandomModel.rng = np.random.default_rng(seed + 2)


def play_one_game(
    agents: dict[chess.Color, RandomAgent | StandardAgent],
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
    agent_kind: str,
    strain: int,
    generation: int,
    confidence: float,
    log: bool,
) -> BenchmarkResult:
    """Run warmup and timed benchmark games."""
    white_agent, black_agent = build_agents(
        agent_kind,
        strain=strain,
        generation=generation,
        confidence=confidence,
    )
    agents = {chess.WHITE: white_agent, chess.BLACK: black_agent}

    return collect_benchmark(
        agents=agents,
        games=games,
        warmup_games=warmup_games,
        log=log,
    )


def collect_benchmark(
    *,
    agents: dict[chess.Color, RandomAgent | StandardAgent],
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
        return "n/a"
    return f"{seconds / count * MILLISECONDS_PER_SECOND:.3f} ms"


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

    print("Benchmark complete", file=output)
    print(f"  games:        {result.games}", file=output)
    print(f"  warmups:      {result.warmup_games}", file=output)
    print(f"  total plies:  {total_plies}", file=output)
    print(f"  outcomes:     {outcomes}", file=output)
    print(f"  total time:   {result.elapsed_seconds:.3f} s", file=output)
    print(f"  per game:     {format_rate(result.elapsed_seconds, result.games)}", file=output)
    print(f"  per ply:      {format_rate(result.elapsed_seconds, total_plies)}", file=output)

    if result.results:
        print(
            f"  game plies:   median={statistics.median(plies):.1f}, max={max(plies)}",
            file=output,
        )
        print(
            "  game time:    "
            f"median={statistics.median(game_seconds) * MILLISECONDS_PER_SECOND:.3f} ms, "
            f"max={max(game_seconds) * MILLISECONDS_PER_SECOND:.3f} ms",
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
        strain=args.strain,
        generation=args.generation,
        confidence=args.confidence,
        log=args.log,
    )

    if args.profile_output is not None:
        profiler.dump_stats(args.profile_output)

    return result


def main() -> None:
    """Run the benchmark CLI."""
    args = parse_args()

    if args.seed is not None:
        seed_random_generators(args.seed)

    if args.profile == "cprofile":
        result = run_profiled(args)
    else:
        result = run_games(
            games=args.games,
            warmup_games=args.warmup_games,
            agent_kind=args.agent,
            strain=args.strain,
            generation=args.generation,
            confidence=args.confidence,
            log=args.log,
        )

    print_summary(result)


if __name__ == "__main__":
    main()
