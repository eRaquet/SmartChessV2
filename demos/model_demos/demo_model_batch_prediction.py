"""Script to demo the model module's prediction tool kit."""

import time

import numpy as np

from modules.model import StandardModel

if __name__ == "__main__":
    # generate rng for random data
    rng = np.random.default_rng()
    encodings = rng.integers(0, 2, (1000, 8, 8, 18), dtype=np.uint8)

    # load model (strain: 0, generation: 0)
    # construct=True if model does not yet exist (for seeding)
    model = StandardModel(0, 0)

    # evaluate random encodings as a batch
    t = time.perf_counter()
    predictions = model.predict_batch(encodings)
    dt = time.perf_counter() - t
    print(f"Evaluated in {dt:.3f} seconds with an average of {dt * 3:.1f} ms per board encoding.")
