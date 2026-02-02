import argparse
from motion import create_brownian_path
from tqdm import trange
import matplotlib.pyplot as plt


def prove_length_inf(T: float, times_to_vary: int):
    """
    small function which calls the previously defined brownian motion with rapidly decreasing delta_t hyperparameters

    args:
        T: float -> upper bound for which to discretize space for (passed to old function)
        times_to_vary: int -> how many base 3 delta t discretizations to make.

    returns:
        bs: list -> list of lengths of items of brownian motions at each timestep
        delta: list -> list of delta_t hyperparameters at each timestep

    Note that the loop is wrapped in tqdm's trange function, which provides a nice progress bar.
    """

    # lists to hold brownian motion lengths at timesteps and delta_ts
    bs, deltas = [], []

    # for loop (incrementing by 1 to avoid div by 0 error) which increments base 3 discretization hyperparameters
    for delta in trange(1, times_to_vary):
        delta_t = 1 / (3**delta)
        # uses previously declared function to create 1D brownian motion with newly created delta_t hyperparameter.
        t, b = create_brownian_path(T, delta_t)
        # appends new brownian motion to array
        bs.append(len(b))
        # appends new delta t to array
        deltas.append(delta_t)

    # check to ensure the same amount of timesteps passed for both vars
    assert len(bs) == len(deltas)

    # returns both for plotting
    return bs, deltas


def plot_lens(bs: list, deltas: list):
    """
    plots len of brownian motion at each delta_t discretized timestep with increasingly smaller delta ts


    args:

        bs: list -> list of brownian motion lengths. should go to "infinity". have a stop condition for its creation
        deltas: list -> list of delta_t hyperparameters.

    returns:

        None

        saves plot to local disk
    """
    plt.figure(figsize=(10, 5))
    plt.plot(bs, deltas)
    plt.title(rf"Plot of the length of brownian motion vs. ($\Delta t$)")
    plt.xlabel("length of $B_t$")
    plt.ylabel(r"$\Delta t$(scaled by base 3)")
    plt.grid(True)
    plt.savefig("length_proof.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--T", type=float, default=1, help="upper bound of timesteps to look at"
    )

    parser.add_argument(
        "--times_to_vary",
        type=int,
        default=10,
        help="how many times to decrease the step size",
    )
    args = parser.parse_args()

    bs, deltas = prove_length_inf(args.T, args.times_to_vary)
    plot_lens(bs, deltas)
