import argparse
from motion import create_brownian_path, plot_path

def prove_length_inf(T: float, times_to_vary: int):
    for delta in range(1, times_to_vary ):
        delta_t = 1/(10 ** delta)
        t, b = create_brownian_path(T, delta_t)
        plot_path(t, b, delta_t)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--T",
        type = float, 
        default = 1,
        help = "upper bound of timesteps to look at"
    )

    parser.add_argument(
        "--times_to_vary",
        type = int,
        default  = 15,
        help = "how many times to decrease the step size"
    )
    args = parser.parse_args()

    prove_length_inf(args.T, args.times_to_vary)
