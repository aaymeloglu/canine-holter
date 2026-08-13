import os
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass
from tkinter import filedialog, messagebox
from canine_holter.pipeline import run_analysis


@dataclass
class AnalysisResult:
    success: bool
    report_path: str | None
    error_message: str | None


def analyze_and_report(input_path: str, out_dir: str) -> AnalysisResult:
    """Runs the pipeline and captures success/failure as data, so this is
    testable without a display and reusable by both the GUI and CLI."""
    try:
        report_path = run_analysis(input_path, out_dir)
        return AnalysisResult(success=True, report_path=report_path, error_message=None)
    except Exception as exc:  # noqa: BLE001 - surfacing any failure to the GUI is the point
        return AnalysisResult(success=False, report_path=None, error_message=str(exc))


def _open_in_default_app(path: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    else:
        subprocess.run(["xdg-open", path], check=False)


def _on_pick_file() -> None:
    input_path = filedialog.askopenfilename(title="Select a Holter recording (.hea file)")
    if not input_path:
        return
    # WFDB records are referenced by their base path (no extension)
    base_path = os.path.splitext(input_path)[0]
    out_dir = filedialog.askdirectory(title="Select an output folder for the report")
    if not out_dir:
        return

    result = analyze_and_report(base_path, out_dir)
    if result.success:
        messagebox.showinfo("Done", f"Report written to {result.report_path}")
        _open_in_default_app(result.report_path)
    else:
        messagebox.showerror("Analysis failed", result.error_message)


def main() -> None:
    root = tk.Tk()
    root.title("Canine Holter Analyzer")
    root.geometry("360x160")

    label = tk.Label(root, text="Select a Holter recording to analyze", pady=20)
    label.pack()

    button = tk.Button(root, text="Choose Recording...", command=_on_pick_file, padx=20, pady=10)
    button.pack()

    root.mainloop()


if __name__ == "__main__":
    main()
