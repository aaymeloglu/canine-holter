import os
import tempfile
from canine_holter.gui import app as gui_app
from canine_holter.gui.app import AnalysisResult, analyze_and_report

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "fixtures")


def test_analyze_and_report_returns_report_path_on_success():
    input_path = os.path.join(FIXTURES_DIR, "mitdb_119", "119")
    with tempfile.TemporaryDirectory() as out_dir:
        result = analyze_and_report(input_path, out_dir)
        assert result.success is True
        assert os.path.exists(result.report_path)


def test_analyze_and_report_reports_failure_on_bad_input():
    with tempfile.TemporaryDirectory() as out_dir:
        result = analyze_and_report("/nonexistent/path/nope", out_dir)
        assert result.success is False
        assert result.error_message


# --- _on_pick_file: exercises the dialog-driven branching logic without a
# real display, by monkeypatching the tkinter dialog/messagebox functions
# that _on_pick_file calls as module-level names in canine_holter.gui.app.


def test_on_pick_file_returns_early_when_file_dialog_cancelled(monkeypatch):
    events = []

    # Tkinter's askopenfilename returns "" (not None) when the user cancels.
    monkeypatch.setattr(gui_app.filedialog, "askopenfilename", lambda **kw: "")
    monkeypatch.setattr(
        gui_app.filedialog,
        "askdirectory",
        lambda **kw: events.append("askdirectory") or "/tmp/should-not-be-used",
    )
    monkeypatch.setattr(
        gui_app, "analyze_and_report", lambda *a, **kw: events.append(("analyze_and_report", a, kw))
    )
    monkeypatch.setattr(gui_app.messagebox, "showinfo", lambda *a, **kw: events.append(("showinfo", a)))
    monkeypatch.setattr(gui_app.messagebox, "showerror", lambda *a, **kw: events.append(("showerror", a)))
    monkeypatch.setattr(gui_app, "_open_in_default_app", lambda *a, **kw: events.append(("open", a)))

    gui_app._on_pick_file()

    # Cancelling the file picker must short-circuit before the output-folder
    # dialog, the pipeline, or any result dialog runs.
    assert events == []


def test_on_pick_file_returns_early_when_output_dir_dialog_cancelled(monkeypatch):
    events = []

    monkeypatch.setattr(gui_app.filedialog, "askopenfilename", lambda **kw: "/some/record.hea")
    # Tkinter's askdirectory returns "" (not None) when the user cancels.
    monkeypatch.setattr(gui_app.filedialog, "askdirectory", lambda **kw: "")
    monkeypatch.setattr(
        gui_app, "analyze_and_report", lambda *a, **kw: events.append(("analyze_and_report", a, kw))
    )
    monkeypatch.setattr(gui_app.messagebox, "showinfo", lambda *a, **kw: events.append(("showinfo", a)))
    monkeypatch.setattr(gui_app.messagebox, "showerror", lambda *a, **kw: events.append(("showerror", a)))
    monkeypatch.setattr(gui_app, "_open_in_default_app", lambda *a, **kw: events.append(("open", a)))

    gui_app._on_pick_file()

    # Cancelling the output-folder picker must short-circuit before the
    # pipeline runs or any result dialog shows.
    assert events == []


def test_on_pick_file_success_passes_selected_recording_and_shows_info(monkeypatch):
    events = []
    captured_args = {}

    monkeypatch.setattr(gui_app.filedialog, "askopenfilename", lambda **kw: "/recordings/119.hea")
    monkeypatch.setattr(gui_app.filedialog, "askdirectory", lambda **kw: "/tmp/out")

    def fake_analyze_and_report(input_path, out_dir):
        captured_args["input_path"] = input_path
        captured_args["out_dir"] = out_dir
        return AnalysisResult(success=True, report_path="/tmp/out/report.pdf", error_message=None)

    monkeypatch.setattr(gui_app, "analyze_and_report", fake_analyze_and_report)
    monkeypatch.setattr(gui_app.messagebox, "showinfo", lambda *a, **kw: events.append(("showinfo", a)))
    monkeypatch.setattr(gui_app.messagebox, "showerror", lambda *a, **kw: events.append(("showerror", a)))
    monkeypatch.setattr(gui_app, "_open_in_default_app", lambda path: events.append(("open", path)))

    gui_app._on_pick_file()

    # Format normalization belongs to the shared ingest dispatcher, so the GUI
    # passes through both WFDB headers and DR200 channel files unchanged.
    assert captured_args == {"input_path": "/recordings/119.hea", "out_dir": "/tmp/out"}
    assert events == [
        ("showinfo", ("Done", "Report written to /tmp/out/report.pdf")),
        ("open", "/tmp/out/report.pdf"),
    ]


def test_on_pick_file_failure_shows_error_and_does_not_open(monkeypatch):
    events = []

    monkeypatch.setattr(gui_app.filedialog, "askopenfilename", lambda **kw: "/recordings/bad.hea")
    monkeypatch.setattr(gui_app.filedialog, "askdirectory", lambda **kw: "/tmp/out")
    monkeypatch.setattr(
        gui_app,
        "analyze_and_report",
        lambda *a, **kw: AnalysisResult(success=False, report_path=None, error_message="boom"),
    )
    monkeypatch.setattr(gui_app.messagebox, "showinfo", lambda *a, **kw: events.append(("showinfo", a)))
    monkeypatch.setattr(gui_app.messagebox, "showerror", lambda *a, **kw: events.append(("showerror", a)))
    monkeypatch.setattr(gui_app, "_open_in_default_app", lambda *a, **kw: events.append(("open", a)))

    gui_app._on_pick_file()

    # On failure, only the error dialog fires - no report is opened.
    assert events == [("showerror", ("Analysis failed", "boom"))]
