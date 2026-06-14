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

# board gui dimentions
BOARD_RIM_THICKNESS = 20
BOARD_WIDTH = 480
SQUARE_WIDTH = BOARD_WIDTH / 8

### Agent Config

# default confidence level
DEFAULT_CONFIDENCE = 8.0
