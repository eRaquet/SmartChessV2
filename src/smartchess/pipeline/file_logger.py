"""Print logger for printing game logs to a text file."""

from dataclasses import dataclass
from typing import override

from smartchess.config import PROJECT_PATH
from smartchess.types import GameLog
from smartchess.util.ui import write_game

from .logger_base import LoggerBase, LoggerConfig


@dataclass(slots=True, kw_only=True)
class FileLoggerConfig(LoggerConfig):
    """Config for print logger."""

    target_path: str
    """
    Relative path to write game log to, from `artifacts/text_logs/`.
    """


class FileLogger(LoggerBase):
    """Print logger that writes game log to file."""

    def __init__(self, config: FileLoggerConfig) -> None:
        self._target_path = PROJECT_PATH / 'artifacts' / 'text_logs' / config.target_path

    @override
    def log_game(self, log: GameLog) -> None:
        write_game(log, self._target_path)
