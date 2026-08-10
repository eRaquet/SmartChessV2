"""Game subapp of smartchess CLI."""

from __future__ import annotations

from typing import Annotated, Any, cast

import typer

from smartchess.agent import create_agent
from smartchess.application import BenchmarkGameConfig, PlayConfig, benchmark_game
from smartchess.application import play as run_play
from smartchess.board import DEFAULT_BOARD_CONFIG, BoardConfig
from smartchess.config import DEFAULT_CONFIDENCE
from smartchess.game import StandardGameConfig
from smartchess.pipeline import FileLoggerConfig, PrintLoggerConfig
from smartchess.ui import GUIConfig, create_gui

from .agent import AgentKind, AgentSpec, LogMode, complete_agent, create_agent_config, parse_agent
from .util import UNSET, Unset, parse_confidence

game_app = typer.Typer(name='game', no_args_is_help=True, help='Manage smartchess game play.')


@game_app.command(no_args_is_help=True)
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


@game_app.command(no_args_is_help=True)
def play(
    white: Annotated[
        AgentSpec,
        typer.Argument(
            parser=parse_agent,
            autocompletion=complete_agent,
            metavar='WHITE',
            help='White agent: ui, random, random-model, or model:STRAIN[:GENERATION].',
        ),
    ],
    black: Annotated[
        AgentSpec,
        typer.Argument(
            parser=parse_agent,
            autocompletion=complete_agent,
            metavar='BLACK',
            help='Black agent: ui, random, random-model, or model:STRAIN[:GENERATION].',
        ),
    ],
    gui: Annotated[
        bool,
        typer.Option(
            '--gui/--no-gui',
            help='Visualize the game in the graphical interface.',
        ),
    ] = False,
    log: Annotated[
        LogMode,
        typer.Option(
            '--log',
            case_sensitive=False,
            help='Game output destination.',
        ),
    ] = LogMode.NONE,
    log_file: Annotated[
        str | None,
        typer.Option(
            '--log-file',
            metavar='str',
            help='Output file name when --log=file, stored in `artifacts/text_logs/`.',
        ),
    ] = None,
    confidence: Annotated[
        float | None,
        typer.Option(
            '--confidence',
            parser=parse_confidence,
            metavar='float | (float, float)',
            help=(
                'Confidence of model-based agent when selected.'
                '  If a field is "inf", than complete confidence is used.'
            ),
            show_default=f'{DEFAULT_CONFIDENCE}',
        ),
    ] = cast('float | None', UNSET),
) -> None:
    """Play one chess game with the selected agents."""
    agent_kinds = {white.kind, black.kind}
    if not gui and AgentKind.UI in agent_kinds:
        msg = 'UI agents require --gui.'
        raise typer.BadParameter(msg, param_hint='--gui')

    if log is LogMode.FILE and log_file is None:
        msg = '--log-file is required when --log=file.'
        raise typer.BadParameter(msg, param_hint='--log-file')

    if log is not LogMode.FILE and log_file is not None:
        msg = '--log-file can only be used with --log=file.'
        raise typer.BadParameter(msg, param_hint='--log-file')

    interface: Any = None
    if gui:
        gui_config = GUIConfig()
        interface = create_gui(gui_config)

    match log:
        case LogMode.NONE:
            output = None
        case LogMode.PRINT:
            output = PrintLoggerConfig()
        case LogMode.FILE:
            if log_file is None:
                msg = 'File logger config requested without a target path.'
                raise RuntimeError(msg)
            output = FileLoggerConfig(target_path=log_file)
        case _:
            msg = 'Unsupported log type.'
            raise ValueError(msg)

    white_confidence: float | None | Unset = (
        confidence if not isinstance(confidence, tuple) else confidence[0]
    )
    black_confidence: float | None | Unset = (
        confidence if not isinstance(confidence, tuple) else confidence[1]
    )
    config = PlayConfig(
        game=StandardGameConfig(
            white=create_agent_config(white, interface, white_confidence),
            black=create_agent_config(black, interface, black_confidence),
            board=BoardConfig(),
            renderer=interface,
            collect=output is not None,
        ),
        output=output,
    )

    try:
        run_play(config)
    finally:
        if interface is not None:
            interface.exit()
