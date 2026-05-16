"""Script to demo the model module's prediction tool kit."""

import time

import numpy as np
from matplotlib import pyplot as plt

from modules.model import StandardModel

if __name__ == "__main__":
    num_iterations = 10

    # generate rng for random data
    rng = np.random.default_rng()

    # load model (strain: 0, generation: 0)
    # construct=True if model does not yet exist (for seeding)
    model = StandardModel(0, 0)

    N = np.array([10, 30, 100, 300, 1000, 3000, 10000])
    times = np.zeros(N.shape)

    for _ in range(num_iterations):
        for i in range(len(N)):
            encodings = rng.integers(0, 2, (N[i], 8, 8, 18), dtype=np.uint8)

            # evaluate random encodings as a batch
            t = time.perf_counter()
            predictions = model.predict_batch(encodings)
            dt = time.perf_counter() - t
            times[i] += dt / N[i] * 1000  # time in miliseconds per position
            print(f"{N[i]} positions evaluated in {dt:.3f} seconds with an average of {dt / N[i] * 1000:.1f} ms per board encoding.")

    plt.figure()
    plt.plot(np.log10(N), np.log10(times / num_iterations))
    plt.xlabel("# of Positions (log scale)")
    plt.ylabel("Time in ms per Position (log scale)")
    plt.title("Model Performance vs. Size of Batch")
    plt.show()
