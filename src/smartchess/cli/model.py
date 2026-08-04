"""Model subapp of CLI."""

from pathlib import Path
from typing import Annotated

import typer

from smartchess.application import SeedConfig
from smartchess.application import seed as run_seed
from smartchess.config import PROJECT_PATH

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
