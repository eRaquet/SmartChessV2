"""Run a benchmark on the model inference performance."""

import time
from typing import TYPE_CHECKING, cast

import numpy as np

from smartchess.capabilities import require_capability
from smartchess.config import PROJECT_PATH
from smartchess.model import ModelConfig, create_model

from .config import BenchmarkModelConfig

if TYPE_CHECKING:
    from smartchess.types import SetEncoding


def benchmark_model(config: BenchmarkModelConfig) -> None:
    """

    Benchmark the performance of a model.

    Parameters
    ----------
    config : BenchmarkModelConfig
        Config for model benchmark.
    """
    require_capability('plot')

    from matplotlib import pyplot as plt  # noqa: PLC0415

    rng = np.random.default_rng()

    if isinstance(config.model, ModelConfig):
        config.model = create_model(config.model)

    if config.log_samples:
        len_encodings = np.linspace(
            np.log10(config.num_encoding_min),
            np.log10(config.num_encoding_max),
            num=config.num_points,
        )
        len_encodings = (10**len_encodings).astype(np.uint64)
    else:
        len_encodings = np.linspace(
            config.num_encoding_min,
            config.num_encoding_max,
            config.num_points,
            dtype=np.uint64,
        )

    times = np.zeros(config.num_points)

    for i in range(config.num_points):
        for _ in range(config.num_trials):
            encodings = cast(
                'SetEncoding', rng.integers(0, 2, (len_encodings[i], 8, 8, 18), dtype=np.uint8)
            )

            start = time.time_ns()

            __ = config.model.predict_batch(encodings)

            stop = time.time_ns()

            times[i] += stop - start

    plt.figure()

    if config.log_graph:
        plt.loglog(len_encodings, times / (len_encodings * config.num_trials))
        plt.xlabel('# of Encodings Per Eval (log scale)')
        plt.ylabel('Mean Time Per Encoding in ns (log scale)')
    else:
        plt.plot(len_encodings, times / (len_encodings * config.num_trials))
        plt.xlabel('# of Encodings Per Eval')
        plt.ylabel('Mean Time Per Encoding in ns')

    plt.title('Model Performance vs. Size of Batch')

    if config.file is not None:
        path = PROJECT_PATH / 'artifacts' / 'plots'
        path.mkdir(exist_ok=True)
        plt.savefig(
            path / config.file,
        )

    plt.show()
