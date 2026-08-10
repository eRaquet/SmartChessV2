"""UI Module."""

from __future__ import annotations

from typing import TYPE_CHECKING

from smartchess.capabilities import require_capability

from .config import GUIConfig as GUIConfig

if TYPE_CHECKING:
    from .gui import GUI


def create_gui(config: GUIConfig) -> GUI:
    """

    Create a GUI from the provided config.

    Parameters
    ----------
    config : GUIConfig
        config for the GUI

    Returns
    -------
    GUI
        created GUI object
    """
    if type(config) is GUIConfig:
        require_capability('gui')

        from .gui import GUI

        return GUI(config)
    msg = 'Unsupported GUI config type'
    raise ValueError(msg)
