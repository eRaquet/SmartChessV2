"""Module to specify the behavior of a chess board inference model."""

import numpy as np

from modules.chess_types import BoardEncoding, Evaluation, SetEncoding, SetEvaluation


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
