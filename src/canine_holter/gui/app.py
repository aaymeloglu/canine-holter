import os
import subprocess
import sys
import threading
import tkinter as tk
from dataclasses import dataclass, replace
from tkinter import filedialog
from canine_holter.pipeline import run_analysis


@dataclass
class AnalysisResult:
    success: bool
    report_path: str | None
    error_message: str | None


def analyze_and_report(input_path: str, out_dir: str) -> AnalysisResult:
    """Run the pipeline and return a result the GUI can display."""
    try:
        report_path = run_analysis(input_path, out_dir)
        return AnalysisResult(success=True, report_path=report_path, error_message=None)
    except Exception as exc:  # noqa: BLE001 - surfacing any failure to the GUI is the point
        return AnalysisResult(success=False, report_path=None, error_message=str(exc))


def _open_in_default_app(path: str) -> None:
    if sys.platform == "darwin":
        subprocess.run(["open", path], check=False)
    elif sys.platform == "win32":
        os.startfile(path)  # the user's own report, in their default PDF viewer
    else:
        subprocess.run(["xdg-open", path], check=False)


# --- State and transitions (no tkinter widgets; the dialogs are called by
# module-level name so tests can monkeypatch them) --------------------------


@dataclass(frozen=True)
class AppState:
    recording_path: str | None = None
    out_dir: str | None = None
    running: bool = False
    result: AnalysisResult | None = None

    @property
    def can_run(self) -> bool:
        return bool(self.recording_path and self.out_dir) and not self.running


def choose_recording(state: AppState) -> AppState:
    path = filedialog.askopenfilename(
        title="Select a Holter recording",
        filetypes=[
            ("Supported recordings", "*.hea *.dat"),
            ("WFDB headers", "*.hea"),
            ("DR200/DR400 flash.dat", "*.dat"),
            ("All files", "*"),
        ],
    )
    if not path:  # cancelled: tkinter returns ""
        return state
    return replace(state, recording_path=path, result=None)


def choose_output(state: AppState) -> AppState:
    path = filedialog.askdirectory(title="Select the folder to write report.pdf into")
    if not path:
        return state
    return replace(state, out_dir=path, result=None)


def run(state: AppState) -> AppState:
    """Run the analysis synchronously for the current choices; a no-op
    unless both are chosen."""
    if not state.can_run:
        return state
    result = analyze_and_report(state.recording_path, state.out_dir)
    return replace(state, running=False, result=result)


def recording_label_text(state: AppState) -> str:
    return os.path.basename(state.recording_path) if state.recording_path else "No recording chosen"


def output_label_text(state: AppState) -> str:
    return state.out_dir or "No output folder chosen"


def status_text(state: AppState) -> str:
    if state.running:
        return "Analyzing... this can take a minute."
    if state.result is not None:
        if state.result.success:
            return f"Done - report written to {state.result.report_path}"
        return f"Analysis failed: {state.result.error_message}"
    if state.can_run:
        return "Ready to run."
    return "Choose a recording and an output folder, then Run."


# --- Window ------------------------------------------------------------------

_POLL_MS = 100


class AnalyzerWindow:
    """Three steps top to bottom: choose recording, choose output folder,
    Run. The analysis runs on a worker thread so the window stays
    responsive, and the status line doubles as the done indicator."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.state = AppState()
        self._pending: AppState | None = None

        root.title("Canine Holter Analyzer")
        root.resizable(False, False)
        frame = tk.Frame(root, padx=20, pady=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1, minsize=360)

        self.recording_button = tk.Button(frame, text="1. Choose recording...", width=22, command=self._pick_recording)
        self.recording_label = tk.Label(frame, anchor="w", fg="#555")
        self.output_button = tk.Button(frame, text="2. Choose output folder...", width=22, command=self._pick_output)
        self.output_label = tk.Label(frame, anchor="w", fg="#555")
        self.run_button = tk.Button(frame, text="3. Run analysis", width=22, command=self._run)
        self.status_label = tk.Label(frame, anchor="w", wraplength=360, justify="left")
        self.open_button = tk.Button(frame, text="Open report", command=self._open_report)

        self.recording_button.grid(row=0, column=0, sticky="w", pady=4)
        self.recording_label.grid(row=0, column=1, sticky="w", padx=(12, 0))
        self.output_button.grid(row=1, column=0, sticky="w", pady=4)
        self.output_label.grid(row=1, column=1, sticky="w", padx=(12, 0))
        self.run_button.grid(row=2, column=0, sticky="w", pady=(12, 4))
        self.status_label.grid(row=2, column=1, sticky="w", padx=(12, 0), pady=(12, 4))
        self.open_button.grid(row=3, column=1, sticky="w", padx=(12, 0))
        self._refresh()

    def _set(self, state: AppState) -> None:
        self.state = state
        self._refresh()

    def _refresh(self) -> None:
        s = self.state
        self.recording_label.config(text=recording_label_text(s))
        self.output_label.config(text=output_label_text(s))
        self.status_label.config(text=status_text(s))
        self.run_button.config(state="normal" if s.can_run else "disabled")
        self.recording_button.config(state="disabled" if s.running else "normal")
        self.output_button.config(state="disabled" if s.running else "normal")
        done = s.result is not None and s.result.success
        self.open_button.config(state="normal" if done else "disabled")

    def _pick_recording(self) -> None:
        self._set(choose_recording(self.state))

    def _pick_output(self) -> None:
        self._set(choose_output(self.state))

    def _run(self) -> None:
        if not self.state.can_run:
            return
        self._set(replace(self.state, running=True, result=None))
        threading.Thread(target=self._worker, daemon=True).start()
        self.root.after(_POLL_MS, self._poll)

    def _worker(self) -> None:
        self._pending = run(replace(self.state, running=False))

    def _poll(self) -> None:
        if self._pending is None:
            self.root.after(_POLL_MS, self._poll)
            return
        finished, self._pending = self._pending, None
        self._set(finished)
        if finished.result is not None and finished.result.success:
            _open_in_default_app(finished.result.report_path)

    def _open_report(self) -> None:
        if self.state.result is not None and self.state.result.success:
            _open_in_default_app(self.state.result.report_path)


def main() -> None:
    root = tk.Tk()
    AnalyzerWindow(root)
    root.mainloop()


if __name__ == "__main__":
    main()
