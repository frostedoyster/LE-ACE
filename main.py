from LE_ACE import run_fit
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="?"
    )

    parser.add_argument(
        "parameters",
        type=str,
        help="The file containing the parameters. JSON formatted dictionary.",
    )

    parser.add_argument(
        "n_train",
        type=int,
        help="The file containing the parameters. JSON formatted dictionary.",
    )

    parser.add_argument(
        "random_seed",
        type=int,
        help="The file containing the parameters. JSON formatted dictionary.",
    )

    args = parser.parse_args()
    run_fit(args.parameters, args.n_train, args.random_seed)
