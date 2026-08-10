"""Helper Functions for Math."""

import numpy as np

from smartchess.types import PMF, SetEvaluation


def entropy(dist: PMF) -> float:
    """

    Calculate the Shannon entropy from the provided distribution.

    Parameters
    ----------
    dist : PMF

    Returns
    -------
    float
        returned Shannon entropy, or None if not applicable
    """
    positive = dist > 0

    # adding 0.0 is to keep the entropy positive, in the case where the entropy is 0
    return float(-np.dot(dist[positive], np.log2(dist[positive]))) + 0.0


def softmax(evals: SetEvaluation) -> PMF:
    """

    Calculate the softmax of the given set of evaluations.

    Parameters
    ----------
    evals : SetEvaluation
        evaluations that have been scaled appropriately

    Returns
    -------
    PMF
        output PMF
    """
    out = np.exp(evals)
    out /= np.sum(out)
    return out
