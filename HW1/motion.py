import argparse
import matplotlib.pyplot as plt
import numpy as np


def create_brownian_path(T: float, delta_t: float):
    """
    creates a sample brownian path

    args:
        T: float -> determines the amount of time for which to simulate brownian motion.
        delta_t: float -> discretization factor determining how many "pieces" to break the interval into.

    Returns:
        steps: np.ndarray -> discretized timesteps from 0 to T
        values: np.ndarray -> values of the brownian motion over the range of steps
    """

    # number of steps based on discretization hyperparameter and duration provided
    steps = int(T / delta_t)
    # print(steps)

    # array corresponding to all of the timesteps: [0, delta_t, 2 * delta_t, 3 * delta_t, etc...]
    timesteps = np.linspace(0, T, steps + 1)

    # sqrt(n) * Z ~ (0, 1). variance of increment over delta_t is delta_t. note that np.random.normal by default assigns mean = 0, sigma = 1
    kicks = np.sqrt(delta_t) * np.random.normal(size=steps)

    # vectorized representation of the array meant to held all of the brownian paths.
    vals = np.zeros(steps + 1)

    # from the 1st idx (not B_0 since its assumed to be 0) , apply the sumsum of sqrt(n) * Z ~(0, 1), or the sum of all actions taken up to that point.
    vals[1:] = np.cumsum(kicks)

    return timesteps, vals


def plot_path(
    timesteps: np.ndarray, brownian_values: np.ndarray, discretization_step_size: float
) -> None:
    plt.figure(figsize=(12, 5))
    plt.step(timesteps, brownian_values)
    plt.title(rf"Brownian Motion ($\Delta t = {discretization_step_size}$)")
    plt.xlabel("Time (t)")
    plt.ylabel("B(t)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"path_{discretization_step_size}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--delta_t",
        type=float,
        default=0.1,
        help="discretization step size factor for brownian motion",
    )
    parser.add_argument("--T", type=float, default=1.0, help="total time")

    args = parser.parse_args()

    rng = np.random.default_rng(67)

    t, b = create_brownian_path(args.T if args.T else 1.0, args.delta_t)
    plot_path(t, b, args.delta_t)
