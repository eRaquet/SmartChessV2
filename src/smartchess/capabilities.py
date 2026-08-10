"""Define capabilities of smartchess install."""

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Literal


@dataclass(slots=True, kw_only=True)
class Capability:
    """A single required capability."""

    name: str
    """Name of capability."""

    group: str
    """group that enables the capability."""

    distributions: tuple[str, ...]
    """Required packages for capability."""


SupportedCapabilities = Literal['gui', 'plot', 'profile', 'table', 'mlx', 'jax']

_CAPABILITIES: dict[SupportedCapabilities, Capability] = {
    'gui': Capability(
        name='graphical interface',
        group='gui',
        distributions=('pygame',),
    ),
    'plot': Capability(
        name='plotting',
        group='plot',
        distributions=('matplotlib',),
    ),
    'profile': Capability(
        name='profile',
        group='profile',
        distributions=('tuna',),
    ),
    'table': Capability(
        name='table',
        group='table',
        distributions=('tabulate',),
    ),
    'mlx': Capability(
        name='MLX inference backend',
        group='backend-mlx',
        distributions=('keras', 'mlx'),
    ),
    'jax': Capability(
        name='JAX inference backend',
        group='backend-jax',
        distributions=('keras', 'jax'),
    ),
}


class MissingCapabilityError(RuntimeError):
    """Missing the capability to perform the requested feature."""

    def __init__(self, capability: Capability, missing_dists: tuple[str, ...]) -> None:
        self.capability = capability
        self.missing = missing_dists

        packages = f'({", ".join(missing_dists)})'
        super().__init__(
            f'Requested capability "{capability.name}" is unavailable, as smartchess is missing '
            f'the required distributions: {packages}.  '
            f'Install with `uv sync --group {capability.group}.'
        )


def require_capability(name: SupportedCapabilities) -> None:
    """Check if the requested capability is supported."""
    capability = _CAPABILITIES[name]
    missing = tuple(dist for dist in capability.distributions if not _distribution_installed(dist))

    if missing:
        raise MissingCapabilityError(capability, missing)


def _distribution_installed(name: str) -> bool:
    """Check if a particular requested distribution is installed."""
    try:
        version(name)
    except PackageNotFoundError:
        return False
    return True
