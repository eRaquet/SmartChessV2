"""Game subapp of smartchess CLI."""

from typing import Annotated

import typer

from smartchess.agent import create_agent
from smartchess.application import BenchmarkGameConfig, benchmark_game
from smartchess.board import DEFAULT_BOARD_CONFIG
from smartchess.config import DEFAULT_CONFIDENCE
from smartchess.game import StandardGameConfig

from .agent import AgentKind, AgentSpec, complete_agent, create_agent_config, parse_agent

game_app = typer.Typer(name='game', no_args_is_help=True, help='Manage smartchess game play.')


@game_app.command()
def benchmark(
    white: Annotated[
        AgentSpec,
        typer.Argument(
            parser=parse_agent,
            autocompletion=complete_agent,
            metavar='WHITE',
            help='White agent: random, random-model, or model:STRAIN[:GENERATION].',
        ),
    ],
    black: Annotated[
        AgentSpec,
        typer.Argument(
            parser=parse_agent,
            autocompletion=complete_agent,
            metavar='BLACK',
            help='Black agent: random, random-model, or model:STRAIN[:GENERATION].',
        ),
    ],
    trials: Annotated[
        int, typer.Option('--trials', metavar='int', help='Number of trials to run.')
    ],
    warmup: Annotated[
        int, typer.Option('--warmup', metavar='int', help='Number of warmup games to run.')
    ] = 1,
    profile_output: Annotated[
        str | None,
        typer.Option(
            '--profile',
            metavar='str',
            help=(
                'Profile to the provided filename in `artifacts/profiles/`.  '
                'If left unset, no profiling will be performed.'
            ),
        ),
    ] = None,
) -> None:
    """Benchmark game play."""
    if white.kind is AgentKind.UI or black.kind is AgentKind.UI:
        msg = 'Cannot perform benchmark on UI agent.'
        raise typer.BadParameter(msg)

    if trials < 1:
        msg = 'Must run at least one trial to collect benchmark.'
        raise typer.BadParameter(msg)
    if warmup < 0:
        msg = 'Cannot perform negative warmups.'
        raise typer.BadParameter(msg)

    white_config = create_agent_config(white, None, DEFAULT_CONFIDENCE)
    black_config = create_agent_config(black, None, DEFAULT_CONFIDENCE)

    white_agent = create_agent(white_config)
    black_agent = create_agent(black_config)

    game_config = StandardGameConfig(
        white=white_agent,
        black=black_agent,
        board=DEFAULT_BOARD_CONFIG,
        renderer=None,
        collect=True,
    )

    benchmark_config = BenchmarkGameConfig(
        game=game_config,
        trials=trials,
        warmup=warmup,
        profile=profile_output is not None,
        profile_output=profile_output,
    )

    benchmark_game(benchmark_config)
