"""Python file that contains some model configurations parameters."""

import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

KERAS_BACKEND = "mlx"

os.environ["KERAS_BACKEND"] = KERAS_BACKEND

from keras.config import set_dtype_policy  # noqa: E402

set_dtype_policy("mixed_float16")

# parameterize the model
MODEL_PARAMS = {
    # shuffling layer, that doesn't change the width of the board, but adds way more channels
    # (note that padding is "same", not "valid")
    "1": {
        "filters": 256,
        "kernal_size": 3,
        "activation": "relu",
        "padding": "same",
        "data_format": "channels_last",
    },
    # condensing layer, that squishes the width of the board down to a single value across many
    # channels.
    "2": {
        "filters": 128,
        "kernal_size": 3,
        "activation": "relu",
        "padding": "valid",
        "data_format": "channels_last",
    },
    "3": {
        "filters": 64,
        "kernal_size": 3,
        "activation": "relu",
        "padding": "valid",
        "data_format": "channels_last",
    },
    "4": {
        "filters": 32,
        "kernal_size": 4,
        "activation": "relu",
        "padding": "valid",
        "data_format": "channels_last",
    },
}
