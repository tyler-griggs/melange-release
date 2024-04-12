import argparse
from melange.lib.runner import SolverRunner
from melange.solver import MelangeSolver


def main(config_path: str, output_format: str):
    runner = SolverRunner(config_path)
    runner.run()
    runner.export(output_format)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Input arguments
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="melange/config/example.json",
        help="Path to the input configuration file, in json",
    )
    # Output arguments
    parser.add_argument(
        "--output_format",
        "-f",
        default="default",
        choices=["skypilot", "default"],
        help="Output format for the results."
        "By default, the result will saved as a json file."
        "SkyPilot format will generate a yaml file for each GPU type based on an existing template",
    )
    args = parser.parse_args()

    main(args.config, args.output_format)
