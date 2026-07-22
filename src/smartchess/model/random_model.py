"""Random value implementation of Model."""

from typing import override

from numpy.random import default_rng

from smartchess.types import BoardEncoding, Evaluation, SetEncoding, SetEvaluation

from .model_base import ModelBase


class RandomModel(ModelBase):
    """Model that randomly evaluates board positions."""

    _rng = default_rng()

    @override
    def predict(self, encoding: BoardEncoding) -> Evaluation:
        """

        Make a random prediction of the probability of winning from the given position.

        Returns
        -------
        Evaluation
            random evaluation between 0 and 1
        """
        return self._rng.random()

    @override
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
        return self._rng.random(len(encodings))
