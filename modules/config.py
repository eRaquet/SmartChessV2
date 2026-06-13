"""Python file that contains various project-wide configuration features."""

from pathlib import Path

### Project Config

# path object to project directory
PROJECT_PATH = Path(__file__).parent.parent

# number of model strains
STRAIN_COUNT = 4

### Graphics Config

# frames per second for user input
FPS = 30

### Agent Config

# default confidence level
DEFAULT_CONFIDENCE = 8.0
