# ruff: noqa: S101
"""Tests for the model module."""

from typing import TYPE_CHECKING, cast

import numpy as np

from smartchess.model import InferenceModelConfig, RandomModelConfig, create_model

if TYPE_CHECKING:
    from smartchess.types import SetEncoding, SetEvaluation


def test_random_model() -> None:
    """Test for the RandomModel class."""
    model_config = RandomModelConfig(seed=0)
    model = create_model(model_config)
    rng = np.random.default_rng()
    test_data: SetEncoding = cast('SetEncoding', rng.integers(0, 2, (10, 8, 8, 18), dtype=np.uint8))
    test_eval = model.predict(test_data[0])
    test_evals: SetEvaluation = model.predict_batch(test_data)

    assert np.all((test_eval <= 1) and (test_eval >= 0))
    assert np.all((test_evals <= 1) & (test_evals >= 0))


def test_inference_model() -> None:
    """Test the InferenceModel class."""
    model_config = InferenceModelConfig(strain=0)
    model = create_model(model_config)
    rng = np.random.default_rng()
    test_data: SetEncoding = cast('SetEncoding', rng.integers(0, 2, (10, 8, 8, 18), dtype=np.uint8))
    test_eval = model.predict(test_data[0])
    test_evals: SetEvaluation = model.predict_batch(test_data)

    assert np.all((test_eval <= 1) and (test_eval >= 0))
    assert np.all((test_evals <= 1) & (test_evals >= 0))

    assert np.allclose(test_eval, test_evals[0])
