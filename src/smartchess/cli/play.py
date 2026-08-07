"""Typer implementation of the ``smartchess play`` command."""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, cast

import typer

from smartchess.config import DEFAULT_CONFIDENCE

from .agent import AgentKind, AgentSpec, LogMode, complete_agent, create_agent_config, parse_agent
from .util import UNSET, Unset, parse_confidence

if TYPE_CHECKING:
    from smartchess.pipeline import LoggerConfig


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
    _validate_options(white, black, gui=gui, log=log, log_file=log_file, confidence=confidence)
    _execute_play(white, black, gui=gui, log=log, log_file=log_file, confidence=confidence)


def _validate_options(
    white: AgentSpec,
    black: AgentSpec,
    *,
    gui: bool,
    log: LogMode,
    log_file: str | None,
    confidence: float | None | tuple[float | None, float | None],
) -> None:
    """Validate relationships between independently parsed CLI values."""
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


def _execute_play(
    white: AgentSpec,
    black: AgentSpec,
    *,
    gui: bool,
    log: LogMode,
    log_file: str | None,
    confidence: float | None | tuple[float | None, float | None],
) -> None:
    """Build application configurations lazily and run the game."""
    from smartchess.application import PlayConfig
    from smartchess.application import play as run_play
    from smartchess.board import BoardConfig
    from smartchess.game import StandardGameConfig

    interface: Any = None
    if gui:
        from smartchess.ui import GUIConfig, create_gui

        gui_config = GUIConfig()
        interface = create_gui(gui_config)

    output = _create_logger_config(log, log_file)
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


def _create_logger_config(log: LogMode, log_file: str | None) -> LoggerConfig | None:
    """Translate the selected output mode into its logger config."""
    if log is LogMode.NONE:
        return None

    if log is LogMode.PRINT:
        from smartchess.pipeline import PrintLoggerConfig

        return PrintLoggerConfig()

    from smartchess.pipeline import FileLoggerConfig

    if log_file is None:
        msg = 'File logger config requested without a target path.'
        raise RuntimeError(msg)
    return FileLoggerConfig(target_path=log_file)
