"""Module to specify the behavior of a chess board inference model."""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO and WARNING from C++
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # Disable oneDNN messages

import json
from pathlib import Path

import numpy as np
from keras import Input, Model
from keras.layers import BatchNormalization, Conv2D, Dense, Reshape
from keras.models import load_model
from keras.optimizers import Adam

from modules.chess_types import BoardEncoding, Evaluation, SetEncoding, SetEvaluation
from modules.config import MODEL_PARAMS, PROJECT_PATH


class ModelBase:
    """Model base class that specifies structure."""

    def predict(self, encoding: BoardEncoding) -> Evaluation:
        """

        Estimated probability of winning from current position.

        Parameters
        ----------
        encoding : BoardEncoding
            Board to predict on

        Returns
        -------
        Evaluation
            Probability of winning from current position,
            between 0 and 1 normally, unless the model is badly trained.
        """
        raise NotImplementedError

    def predict_batch(self, encodings: SetEncoding) -> SetEvaluation:
        """

        Predict on a set of board encodings.

        Parameters
        ----------
        encodings : SetEncoding
            Encodings to predict for.

        Returns
        -------
        SetEvaluation
            Array of predictions
        """
        raise NotImplementedError


class RandomModel(ModelBase):
    """Model that randomly evaluates board positions."""

    rng = np.random.default_rng()

    def predict(self, _: BoardEncoding) -> Evaluation:
        """

        Make a random prediction of the probability of winning from the given position.

        Returns
        -------
        Evaluation
            random evaluation between 0 and 1
        """
        return self.rng.random()

    def predict_batch(self, encodings: SetEncoding) -> SetEvaluation:
        """

        Make a random prediction of the probability of winning from the given position.

        Parameters
        ----------
        encodings : SetEncoding
            Encodings to predict for.

        Returns
        -------
        Evaluation
            random evaluation between 0 and 1
        """
        return self.rng.random(len(encodings))


class StandardModel(ModelBase):
    """CNN-based evaluation model for chess boards."""

    def __init__(self, strain: int, generation: int, *, construct: bool = False) -> None:
        """

        Create a standard model.

        Requires the saved model `metadata.json` to be configured.

        Parameters
        ----------
        strain : int
            strain number of model
        generation : int
            generation number of model
        construct : bool, optional
            whether to construct the model from scratch and save to memory, or load it from memory,
            by default False
        """
        self._strain = strain
        self._generation = generation

        if construct:
            # create model strain directory if it doesn't already exits
            (PROJECT_PATH / "data" / "saved_models" / f"strain_{self._strain}").mkdir(
                parents=True, exist_ok=True
            )

            # construct model from scratch
            self._construct_model()

            # save constructed model
            self.save(keep_generation=True)

        else:
            self._load()

    def predict(self, encoding: BoardEncoding) -> Evaluation:
        """

        Estimated probability of winning from current position.

        Parameters
        ----------
        encoding : BoardEncoding
            Board to predict on

        Returns
        -------
        Evaluation
            Probability of winning from current position,
            between 0 and 1 normally, unless the model is badly trained.
        """
        # cast encoding to the proper shape and data type
        encoding_recasted = encoding.astype(np.float16).reshape((1, *encoding.shape))

        return self._model.predict_on_batch(encoding_recasted)[0, 0]

    def predict_batch(self, encodings: SetEncoding) -> SetEvaluation:
        """

        Make a random prediction of the probability of winning from the given position.

        Parameters
        ----------
        encodings : SetEncoding
            Encodings to predict for.

        Returns
        -------
        Evaluation
            random evaluation between 0 and 1
        """
        # cast encoding to the proper shape and data type
        encodings_recasted = encodings.astype(np.float16)

        return self._model.predict(encodings_recasted, verbose=0).reshape(
            (len(encodings_recasted),)
        )

    def save(self, *, keep_generation: bool = False, new_generation: bool = False) -> None:
        """

        Save model after training.

        Parameters
        ----------
        keep_generation : bool, optional
            update the current latest generation without upgrading to a new generation, by default
            False
        new_generation : bool, optional
            save the current model as a new generation, by default False
        """
        curr_generation = self.get_curr_generation()

        # check for trying to save new versions of old model generations (not good for record
        # keeping)
        if self._generation != curr_generation:
            msg = "Can only update the most current model \
                generation to maintain backward compatability."
            raise RuntimeError(msg)

        # update current model
        if keep_generation:
            self._model.save(
                PROJECT_PATH
                / "data"
                / "saved_models"
                / f"strain_{self._strain}"
                / f"{self.name}.keras"
            )

        # save model as a new generation
        elif new_generation:
            self._generation += 1
            self.set_curr_generation(self._generation)
            self._model.save(
                PROJECT_PATH
                / "data"
                / "saved_models"
                / f"strain_{self._strain}"
                / f"{self.name}.keras"
            )

        else:
            msg = "Please specify whether to save as a new \
                generation or an update of a past generation."
            raise RuntimeError(msg)

    def _load(self) -> None:
        """Load model from `.keras` file."""
        try:
            self._model = load_model(
                PROJECT_PATH
                / "data"
                / "saved_models"
                / f"strain_{self._strain}"
                / f"{self.name}.keras"
            )
        except ValueError:
            msg = (
                f"Unable to load model: strain {self._strain}"
                f" generation {self._generation} not found."
            )
            raise ValueError(msg) from None

    @property
    def name(self) -> str:
        """

        Get the name of the model.

        Returns
        -------
        str
            name of model
        """
        return f"strain_{self._strain}_gen_{self._generation}"

    def get_curr_generation(self) -> int:
        """

        Get the highest generation number from this model's strain.

        Returns
        -------
        int
            Generation number
        """
        with Path.open(
            PROJECT_PATH / "data" / "saved_models" / "metadata.json", "r"
        ) as metadata_file:
            metadata = json.load(metadata_file)

        return metadata[f"strain_{self._strain}_curr_gen"]

    def set_curr_generation(self, generation_num: int) -> None:
        """

        Set the highest generation number for this model's strain.

        Parameters
        ----------
        generation_num : int
            Generation number to set
        """
        with Path.open(
            PROJECT_PATH / "data" / "saved_models" / "metadata.json", "r"
        ) as metadata_file:
            metadata = json.load(metadata_file)
            metadata[f"strain_{self._strain}_curr_gen"] = generation_num
        with Path.open(
            PROJECT_PATH / "data" / "saved_models" / "metadata.json", "w"
        ) as metadata_file:
            json.dump(metadata, metadata_file)

    def _construct_model(self) -> None:
        """
        Construct a model for this object.

        Uses the configuration parameters in `config.py`, and
        stores in `self._model`.
        """
        # input layer
        input_layer = Input((8, 8, 18), dtype="float16")

        # convolution layers

        temp_layer = Conv2D(
            filters=MODEL_PARAMS["1"]["filters"],
            kernel_size=MODEL_PARAMS["1"]["kernal_size"],
            activation=MODEL_PARAMS["1"]["activation"],
            padding=MODEL_PARAMS["1"]["padding"],
            data_format=MODEL_PARAMS["1"]["data_format"],
        )(input_layer)
        temp_layer = BatchNormalization()(temp_layer)
        temp_layer = Conv2D(
            filters=MODEL_PARAMS["2"]["filters"],
            kernel_size=MODEL_PARAMS["2"]["kernal_size"],
            activation=MODEL_PARAMS["2"]["activation"],
            padding=MODEL_PARAMS["2"]["padding"],
            data_format=MODEL_PARAMS["2"]["data_format"],
        )(temp_layer)
        temp_layer = BatchNormalization()(temp_layer)
        temp_layer = Conv2D(
            filters=MODEL_PARAMS["3"]["filters"],
            kernel_size=MODEL_PARAMS["3"]["kernal_size"],
            activation=MODEL_PARAMS["3"]["activation"],
            padding=MODEL_PARAMS["3"]["padding"],
            data_format=MODEL_PARAMS["3"]["data_format"],
        )(temp_layer)
        temp_layer = BatchNormalization()(temp_layer)
        temp_layer = Conv2D(
            filters=MODEL_PARAMS["4"]["filters"],
            kernel_size=MODEL_PARAMS["4"]["kernal_size"],
            activation=MODEL_PARAMS["4"]["activation"],
            padding=MODEL_PARAMS["4"]["padding"],
            data_format=MODEL_PARAMS["4"]["data_format"],
        )(temp_layer)
        temp_layer = BatchNormalization()(temp_layer)

        # squishing layer
        temp_layer = Dense(1, activation="sigmoid")(temp_layer)
        output_layer = Reshape((1,))(temp_layer)

        # build model
        opt = Adam()
        self._model = Model(inputs=input_layer, outputs=output_layer, name=self.name)
        self._model.compile(optimizer=opt, loss="mean_squared_error")
