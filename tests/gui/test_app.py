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


# --- Three-step flow: choose recording, choose output folder, run. The
# state is a plain dataclass and the transitions are module-level functions
# that call the tkinter dialogs as module-level names, so all of this runs
# without a display; only the widget construction in main() is untested.
import os.path
from canine_holter.gui.app import (
    AppState,
    choose_output,
    choose_recording,
    output_label_text,
    recording_label_text,
    run,
    status_text,
)


def test_initial_state_cannot_run_and_prompts_for_both_choices():
    state = AppState()
    assert state.can_run is False
    assert recording_label_text(state) == "No recording chosen"
    assert output_label_text(state) == "No output folder chosen"
    assert status_text(state) == "Choose a recording and an output folder, then Run."


def test_choose_recording_sets_path_and_label_shows_file_name(monkeypatch):
    monkeypatch.setattr(gui_app.filedialog, "askopenfilename", lambda **kw: "/Volumes/NO NAME/flash.dat")
    state = choose_recording(AppState())
    assert state.recording_path == "/Volumes/NO NAME/flash.dat"
    assert recording_label_text(state) == "flash.dat"


def test_choose_recording_cancel_keeps_previous_choice(monkeypatch):
    # Tkinter's askopenfilename returns "" (not None) when the user cancels.
    monkeypatch.setattr(gui_app.filedialog, "askopenfilename", lambda **kw: "")
    before = AppState(recording_path="/r/119.hea")
    assert choose_recording(before) == before


def test_choose_output_sets_dir_and_label_shows_folder(monkeypatch):
    monkeypatch.setattr(gui_app.filedialog, "askdirectory", lambda **kw: "/Users/andy/Downloads/teeny")
    state = choose_output(AppState())
    assert state.out_dir == "/Users/andy/Downloads/teeny"
    assert output_label_text(state) == "/Users/andy/Downloads/teeny"


def test_choose_output_cancel_keeps_previous_choice(monkeypatch):
    monkeypatch.setattr(gui_app.filedialog, "askdirectory", lambda **kw: "")
    before = AppState(out_dir="/out")
    assert choose_output(before) == before


def test_can_run_only_when_both_chosen_and_not_running():
    assert AppState(recording_path="/r/119.hea").can_run is False
    assert AppState(out_dir="/out").can_run is False
    assert AppState(recording_path="/r/119.hea", out_dir="/out").can_run is True
    assert AppState(recording_path="/r/119.hea", out_dir="/out", running=True).can_run is False


def test_status_text_ready_running_done_failed():
    ready = AppState(recording_path="/r/119.hea", out_dir="/out")
    assert status_text(ready) == "Ready to run."
    assert status_text(AppState(recording_path="/r", out_dir="/out", running=True)) == "Analyzing... this can take a minute."
    done = AppState(recording_path="/r", out_dir="/out",
                    result=AnalysisResult(success=True, report_path="/out/report.pdf", error_message=None))
    assert status_text(done) == "Done - report written to /out/report.pdf"
    failed = AppState(recording_path="/r", out_dir="/out",
                      result=AnalysisResult(success=False, report_path=None, error_message="boom"))
    assert status_text(failed) == "Analysis failed: boom"


def test_choosing_a_new_recording_clears_a_previous_result(monkeypatch):
    monkeypatch.setattr(gui_app.filedialog, "askopenfilename", lambda **kw: "/r/other.hea")
    done = AppState(recording_path="/r/119.hea", out_dir="/out",
                    result=AnalysisResult(success=True, report_path="/out/report.pdf", error_message=None))
    assert choose_recording(done).result is None


def test_run_passes_choices_through_and_records_the_result(monkeypatch):
    captured = {}

    def fake_analyze_and_report(input_path, out_dir):
        captured.update(input_path=input_path, out_dir=out_dir)
        return AnalysisResult(success=True, report_path="/out/report.pdf", error_message=None)

    monkeypatch.setattr(gui_app, "analyze_and_report", fake_analyze_and_report)
    state = run(AppState(recording_path="/r/119.hea", out_dir="/out"))
    assert captured == {"input_path": "/r/119.hea", "out_dir": "/out"}
    assert state.result.report_path == "/out/report.pdf"
    assert state.running is False


def test_run_without_both_choices_is_a_no_op(monkeypatch):
    monkeypatch.setattr(gui_app, "analyze_and_report", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("must not run")))
    before = AppState(recording_path="/r/119.hea")
    assert run(before) == before


def test_open_in_default_app_uses_startfile_on_windows(monkeypatch):
    import sys
    from canine_holter.gui import app as gui

    opened = []
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(gui.os, "startfile", lambda p: opened.append(p), raising=False)
    monkeypatch.setattr(gui.subprocess, "run", lambda *a, **k: opened.append(("subprocess", a)))
    gui._open_in_default_app("C:\\reports\\report.pdf")
    assert opened == ["C:\\reports\\report.pdf"]
