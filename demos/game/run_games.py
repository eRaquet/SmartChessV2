"""Benchmark full game playback."""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
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
from modules.chess_types import AgentDecision, BoardOutcome
from modules.model import RandomModel, StandardModel

NANOSECONDS_PER_SECOND = 1_000_000_000
MILLISECONDS_PER_SECOND = 1_000
DEFAULT_GAMES = 100
DEFAULT_MAX_PLIES = 512
DEFAULT_PROFILE_ROWS = 40


@dataclass(frozen=True, slots=True)
class GameResult:
    """Timing and outcome data for one benchmark game."""

    plies: int
    elapsed_seconds: float
    outcome: BoardOutcome
    capped: bool


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    """Aggregate timing data for a benchmark run."""

    games: int
    warmup_games: int
    max_plies: int
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
        "--max-plies",
        type=int,
        default=DEFAULT_MAX_PLIES,
        help="maximum half-moves per game before the benchmark caps that game",
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
        default=None,
        help="write cProfile stats to this path when --profile cprofile is used",
    )
    parser.add_argument(
        "--profile-rows",
        type=int,
        default=DEFAULT_PROFILE_ROWS,
        help="number of cProfile rows to print",
    )
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


def select_outcome(board: Board, *, capped: bool) -> BoardOutcome:
    """Return the benchmark outcome for a board."""
    if capped:
        return BoardOutcome.ABORT
    if board.winner == chess.WHITE:
        return BoardOutcome.WHITE
    if board.winner == chess.BLACK:
        return BoardOutcome.BLACK
    if board.terminated:
        return BoardOutcome.DRAW
    return BoardOutcome.UNDECIDED


def play_one_game(
    agents: dict[chess.Color, RandomAgent | StandardAgent],
    *,
    max_plies: int,
) -> GameResult:
    """Play and time one benchmark game."""
    start_ns = time.perf_counter_ns()
    board = Board()

    while not board.terminated and board.half_move_count < max_plies:
        decision: AgentDecision = agents[board.turn].act(board)
        board.step(decision.action)

    elapsed_seconds = (time.perf_counter_ns() - start_ns) / NANOSECONDS_PER_SECOND
    capped = not board.terminated
    return GameResult(
        plies=board.half_move_count,
        elapsed_seconds=elapsed_seconds,
        outcome=select_outcome(board, capped=capped),
        capped=capped,
    )


def run_games(
    *,
    games: int,
    warmup_games: int,
    max_plies: int,
    agent_kind: str,
    strain: int,
    generation: int,
    confidence: float,
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
        max_plies=max_plies,
    )


def collect_benchmark(
    *,
    agents: dict[chess.Color, RandomAgent | StandardAgent],
    games: int,
    warmup_games: int,
    max_plies: int,
) -> BenchmarkResult:
    """Run warmup and timed games with already constructed agents."""
    for _ in range(warmup_games):
        play_one_game(
            agents,
            max_plies=max_plies,
        )

    benchmark_start_ns = time.perf_counter_ns()
    results = [
        play_one_game(
            agents,
            max_plies=max_plies,
        )
        for _ in range(games)
    ]
    elapsed_seconds = (time.perf_counter_ns() - benchmark_start_ns) / NANOSECONDS_PER_SECOND

    return BenchmarkResult(
        games=games,
        warmup_games=warmup_games,
        max_plies=max_plies,
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
    capped_games = sum(game.capped for game in result.results)
    outcomes = {
        outcome.name: sum(game.outcome == outcome for game in result.results)
        for outcome in (
            BoardOutcome.WHITE,
            BoardOutcome.BLACK,
            BoardOutcome.DRAW,
            BoardOutcome.ABORT,
        )
    }

    print("Benchmark complete", file=output)
    print(f"  games:        {result.games}", file=output)
    print(f"  warmups:      {result.warmup_games}", file=output)
    print(f"  total plies:  {total_plies}", file=output)
    print(f"  capped games: {capped_games} / {result.games}", file=output)
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
        max_plies=args.max_plies,
        agent_kind=args.agent,
        strain=args.strain,
        generation=args.generation,
        confidence=args.confidence,
    )

    if args.profile_output is not None:
        profiler.dump_stats(args.profile_output)

    stats_stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stats_stream).sort_stats("cumtime")
    stats.print_stats(args.profile_rows)
    print(stats_stream.getvalue())

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
            max_plies=args.max_plies,
            agent_kind=args.agent,
            strain=args.strain,
            generation=args.generation,
            confidence=args.confidence,
        )

    print_summary(result)


if __name__ == "__main__":
    main()
