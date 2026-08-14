# src/canine_holter/cli.py
import argparse
import sys
from canine_holter.pipeline import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="canine-holter",
        description="Analyze a Holter ECG recording for PVC burden and arrhythmias.",
    )
    parser.add_argument(
        "input",
        help=(
            "WFDB record (.hea or base path), native DR200 flash.dat, or decoded "
            "DR200 channel (flashc0.dat/flashc1.dat/flashc2.dat/.raw)"
        ),
    )
    parser.add_argument("--out", required=True, help="Output directory for the report")
    parser.add_argument(
        "--dog-weight-class",
        choices=["small", "medium", "large"],
        default="medium",
        help="Selects brady/tachycardia thresholds (default: medium)",
    )
    args = parser.parse_args()

    report_path = run_analysis(args.input, args.out, dog_weight_class=args.dog_weight_class)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    sys.exit(main())
