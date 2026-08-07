"""Top-level SmartChess command-line application."""

import typer

from .game import game_app
from .model import model_app

app = typer.Typer(
    name='smartchess', no_args_is_help=True, help='Reinforcement Learning Chess Platform'
)

app.add_typer(model_app)
app.add_typer(game_app)


@app.callback()
def root() -> None:
    """Run SmartChess applications."""


def main() -> None:
    """Run Smartchess CLI."""
    app()


if __name__ == '__main__':
    main()
