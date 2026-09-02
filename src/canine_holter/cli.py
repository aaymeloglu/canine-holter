# src/canine_holter/cli.py
import argparse
import sys
from canine_holter.pipeline import StartTimeError, run_analysis


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="canine-holter",
        description="Analyze a Holter ECG recording for PVC burden and arrhythmias.",
    )
    parser.add_argument(
        "input",
        help=(
            "WFDB record (.hea or base path) or native DR200/DR400 flash.dat (any name)"
        ),
    )
    parser.add_argument("--out", required=True, help="Output directory for the report")
    parser.add_argument(
        "--dog-weight-class",
        choices=["small", "medium", "large"],
        default="medium",
        help="Selects brady/tachycardia thresholds (default: medium)",
    )
    parser.add_argument(
        "--start-time",
        help=(
            "Override the recording start time when the recorder's clock is wrong: "
            "HH:MM, HH:MM:SS, or 'YYYY-MM-DD HH:MM[:SS]'. Time-only values keep the "
            "recording's own date."
        ),
    )
    args = parser.parse_args()

    try:
        report_path = run_analysis(
            args.input,
            args.out,
            dog_weight_class=args.dog_weight_class,
            start_time=args.start_time,
        )
    except StartTimeError as exc:
        parser.error(f"--start-time: {exc}")
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    sys.exit(main())
