"""Base Class of Model."""

from abc import ABC, abstractmethod

from smartchess.types import BoardEncoding, Evaluation, SetEncoding, SetEvaluation


class ModelBase(ABC):
    """Model base class that specifies structure."""

    @abstractmethod
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

    @abstractmethod
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
