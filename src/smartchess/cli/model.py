"""Model subapp of CLI."""

from pathlib import Path
from typing import Annotated

import typer

from smartchess.application import BenchmarkModelConfig, SeedConfig
from smartchess.application import benchmark_model as run_benchmark
from smartchess.application import seed as run_seed
from smartchess.cli.util import nonnegative_integer
from smartchess.config import PROJECT_PATH
from smartchess.model import InferenceModelConfig, create_model

model_app = typer.Typer(name='model', no_args_is_help=True, help='Manage smartchess models.')


@model_app.command()
def seed(
    path: Annotated[
        Path,
        typer.Option(
            '--path', metavar='str', help='Path to directory in which to place the seeded models.'
        ),
    ] = PROJECT_PATH / 'artifacts' / 'saved_models',
) -> None:
    """Seed models."""
    config = SeedConfig(path=path)
    run_seed(config)


@model_app.command()
def benchmark(
    strain: Annotated[
        int,
        typer.Option(
            '--strain',
            parser=lambda x: nonnegative_integer(x, 'strain'),
            metavar='int',
            help='Strain of the model to benchmark.',
        ),
    ],
    generation: Annotated[
        int | None,
        typer.Option(
            '--generation',
            parser=lambda x: nonnegative_integer(x, 'generation'),
            metavar='int | None',
            help='Generation of the model to benchmark, if `None` than pick the latest generation.',
        ),
    ] = None,
    encoding_range: Annotated[
        tuple[int, int],
        typer.Option('--range', metavar='(int, int)', help='Range to plot on, [--range min max].'),
    ] = (1, 10000),
    points: Annotated[
        int,
        typer.Option(
            '--points',
            parser=lambda x: nonnegative_integer(x, 'points'),
            metavar='int',
            help='Number of points to plot.',
        ),
    ] = 10,
    log_samples: Annotated[
        bool,
        typer.Option(
            '--log_samples/--linear_samples',
            help='Whether or not to sample logarithmically or linearly.',
        ),
    ] = True,
    log_plot: Annotated[
        bool,
        typer.Option(
            '--log_plot/--linear_plot', help='Whether to plot logarithmically or linearly.'
        ),
    ] = True,
    num_trials: Annotated[
        int,
        typer.Option(
            '--trials',
            metavar='int',
            parser=lambda x: nonnegative_integer(x, 'trials'),
            help='How many trials to run for each point.',
        ),
    ] = 20,
    file: Annotated[
        str | None,
        typer.Option(
            '--file',
            metavar='str',
            help='Filename of output plot, or `None` if no file should be automatically saved.',
        ),
    ] = None,
    path: Annotated[
        Path, typer.Option('--path', metavar='Path', help='Folder where models are located.')
    ] = PROJECT_PATH / 'artifacts' / 'saved_models',
) -> None:
    """Benchmark a smartchess model's performance."""
    model_config = InferenceModelConfig(strain=strain, generation=generation, path=path)
    model = create_model(model_config)

    if encoding_range[0] < 1 or encoding_range[1] < 1 or encoding_range[0] > encoding_range[1]:
        msg = f'Invalid encoding size range {encoding_range}'
        raise typer.BadParameter(msg)

    benchmark_config = BenchmarkModelConfig(
        model=model,
        num_encoding_min=encoding_range[0],
        num_encoding_max=encoding_range[1],
        num_points=points,
        log_samples=log_samples,
        log_graph=log_plot,
        num_trials=num_trials,
        file=file,
    )

    run_benchmark(benchmark_config)
