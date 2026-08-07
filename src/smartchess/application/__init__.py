"""Module that contains applications for smartchess."""

from .benchmark_game import benchmark_game as benchmark_game
from .benchmark_model import benchmark_model as benchmark_model
from .config import BenchmarkGameConfig as BenchmarkGameConfig
from .config import BenchmarkModelConfig as BenchmarkModelConfig
from .config import PlayConfig as PlayConfig
from .config import SeedConfig as SeedConfig
from .play import play as play
from .seed import seed as seed
