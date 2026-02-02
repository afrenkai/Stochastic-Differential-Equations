import numpy as np
import matplotlib.pyplot as plt
from motion import create_brownian_path
import argparse


def compute_derivative(T: float, point: float, delta_t: float):

    """
    Computes the derivative (or attempts to) for some given point using finite difference approximation

    args:

        T: float -> upper bound for which to discretize space for (passed to old function)
        point: float -> the point for which to find the derivative of brownian motion for
        delta_t: float -> the discretization step size for which to create some ground truth brownian motion for
    """

    #little edge case if a user inputs some crazy point since I don't excplicitly handle T >= 1
    if point >= T:
        raise ValueError(
            "the point is equal to or greater than the upper bound of the allowed search space. pick another one"
        )

    #creation of ground truth path
    t, b = create_brownian_path(T, delta_t)

    # discretized (int) step
    idx_t = int(np.round(point / delta_t))

    #same check. not sure if there is EVER a case someone get to this but doesnt hurt to have this just in case
    if idx_t >= len(b):
        raise IndexError(f"Error: target_t {point} is out of bounds for T={T}")


    # super pythonic way to vary exponentiation of base 10, creates 10 ^-1, 10^-2, 10^-3 etc...
    h_values = [10 ** (-p) for p in range(1, 6)]

    #2 blank lists for slopes and valid hs for the derivative calcs. 
    slopes = []
    valid_h = []



    for h in h_values:
        k = int(np.round(h / delta_t)) # idx offset for h 
        idx_h = idx_t + k

        if idx_h < len(b):
            val_t = b[idx_t] #B(t)
            val_th = b[idx_h] #B(t+h)
            slope = (val_th - val_t) / h
            slopes.append(slope)
            valid_h.append(h)
        else:
            #edge case to handle something I missed. making sure we are in the bound of h
            print(f"Skipping h={h}: t+h is out of bounds.")
    return slopes, valid_h


def plot_derivative(chosen_point: float, h: list, slopes: list):
    plt.figure(figsize=(10, 6))
    abs_slopes = np.abs(slopes)
    plt.plot(h, abs_slopes, "o-", linewidth=2, label="Finite Difference Approx")

    #if len h 0, something went wrong
    if len(h) > 0:
        ref_h = np.array(h)
        # plot of 1 / sqrt(h), which is the "actual" growth of this plot
        ref_growth = 1.0 / np.sqrt(ref_h)
        scale_factor = abs_slopes[-1] / ref_growth[-1]
        plt.plot(
            h,
            ref_growth * scale_factor,
            "r--",
            alpha=0.5,
            label=r"Reference $\propto 1/\sqrt{h}$",
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.gca().invert_xaxis()
    plt.xlabel("Step size h (log scale)")
    plt.ylabel("|Approximated Derivative| (log scale)")
    plt.title(f"Non-differentiability of Brownian Motion at t={chosen_point}")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    output_file = "derivative_convergence.png"
    plt.savefig(output_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--point",
        type=float,
        default=0.67,
        help="point in [0,1] to test the derivative for",
    )

    parser.add_argument("--T", type=float, default=1.0, help="total time")

    parser.add_argument(
        "--delta_t",
        type=float,
        default=1e-6,
        help="discretization step size factor for brownian motion",
    )

    args = parser.parse_args()

    slopes, h = compute_derivative(args.T, args.point, args.delta_t)
    plot_derivative(args.point, h, slopes)
