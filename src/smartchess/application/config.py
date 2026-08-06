"""Config for smartchess apps."""

from dataclasses import dataclass
from pathlib import Path

from smartchess.game import GameConfig
from smartchess.model import ModelBase, ModelConfig
from smartchess.pipeline import LoggerConfig


@dataclass(slots=True, kw_only=True)
class PlayConfig:
    """Config for playing a game."""

    game: GameConfig
    """
    Config for game.
    """

    output: LoggerConfig | None = None
    """
    Logger configuration, or None if game is not to be logged.
    """


@dataclass(slots=True, kw_only=True)
class SeedConfig:
    """Config for seeding the models."""

    path: Path
    """
    Path to the directory to seed the models to.
    """


@dataclass(slots=True, kw_only=True)
class BenchmarkModelConfig:
    """Config for benchmarking of a model."""

    model: ModelConfig | ModelBase
    """
    Config for model, or already created model.
    """

    num_encoding_min: int
    """
    The min number of encodings to plot.
    """

    num_encoding_max: int
    """
    The max number of encodings to plot.
    """

    num_points: int
    """
    The number of points to obtain along the graph.
    """

    log_samples: bool
    """
    Whether to draw points logarithmicly or linearly.
    """

    log_graph: bool
    """
    Whether to plot on a log graph.
    """

    num_trials: int
    """
    How many trials to run for each point.
    """

    file: str | None
    """
    Output file name to plot to, or None if no save is desired.

    Plots will be saved in `./artifacts/plots/`.  If None, the plot will only be displayed.
    """
