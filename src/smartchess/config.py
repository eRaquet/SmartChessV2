"""Python file that contains various project-wide configuration features."""

import os
from pathlib import Path

### Project Config

# path object to project directory
PROJECT_PATH = Path(__file__).parent.parent.parent

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

### Model Config

# get environment variables
_backend = os.environ.get('KERAS_BACKEND', default='mlx')
if _backend not in ['jax', 'mlx']:
    msg = 'Only "jax" and "mlx" are supported keras backends'
    raise ValueError(msg)
KERAS_BACKEND = _backend

# set keras backend, if it was not already set
os.environ['KERAS_BACKEND'] = KERAS_BACKEND

KERAS_DTYPE_POLICY = os.environ.get('KERAS_DTYPE_POLICY', 'mixed_float16')

# inference model shape
MODEL_PARAMS = {
    # shuffling layer, that doesn't change the width of the board, but adds way more channels
    # (note that padding is "same", not "valid")
    1: {
        'filters': 256,
        'kernal_size': 3,
        'activation': 'relu',
        'padding': 'same',
        'data_format': 'channels_last',
    },
    # condensing layer, that squishes the width of the board down to a single value across many
    # channels.
    2: {
        'filters': 128,
        'kernal_size': 3,
        'activation': 'relu',
        'padding': 'valid',
        'data_format': 'channels_last',
    },
    3: {
        'filters': 64,
        'kernal_size': 3,
        'activation': 'relu',
        'padding': 'valid',
        'data_format': 'channels_last',
    },
    4: {
        'filters': 32,
        'kernal_size': 4,
        'activation': 'relu',
        'padding': 'valid',
        'data_format': 'channels_last',
    },
}
