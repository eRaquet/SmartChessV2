"""Pipeline Module."""

from .collector import Collector as Collector
from .file_logger import FileLogger as FileLogger
from .file_logger import FileLoggerConfig as FileLoggerConfig
from .logger_base import LoggerBase as LoggerBase
from .logger_base import LoggerConfig as LoggerConfig
from .print_logger import PrintLogger as PrintLogger
from .print_logger import PrintLoggerConfig as PrintLoggerConfig


def create_logger(config: LoggerConfig) -> LoggerBase:
    """

    Create logger from provided config.

    Parameters
    ----------
    config : LoggerConfig
        desired logger config

    Returns
    -------
    LoggerBase
        created logger
    """
    if type(config) is PrintLoggerConfig:
        return PrintLogger(config)
    if type(config) is FileLoggerConfig:
        return FileLogger(config)
    msg = 'Unsupported logger config type.'
    raise ValueError(msg)
