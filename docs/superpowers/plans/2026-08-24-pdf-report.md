# PDF Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Write `report.pdf` (summary text + timeline on page 1, three event strips per page after) and make it the artifact the CLI prints and the GUI opens.

**Architecture:** Figure-drawing code in `timeline.py`/`generate.py` is split into "draw into this axes/region" functions reused by both the PNGs and a new `report/pdf.py` page assembler built on matplotlib `PdfPages`. `write_report` returns the PDF path. Spec: `docs/superpowers/specs/2026-08-24-pdf-report-design.md`.

**Tech Stack:** matplotlib (`PdfPages`, `GridSpec`), pytest. Run tests with `.venv/bin/pytest -q`.

---

### Task 1: `draw_timeline(fig, subplot_spec, ...)` refactor

**Files:** Modify `src/canine_holter/report/timeline.py`; Test `tests/report/test_timeline.py`

- [ ] **Step 1: Failing test**

```python
def test_draw_timeline_draws_into_given_figure_region():
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    from canine_holter.report.timeline import draw_timeline

    fig = plt.figure(figsize=(8.5, 11))
    gs = GridSpec(2, 1, figure=fig)
    beats = [_beat(i * 0.8, 0.8 if i else None, "N") for i in range(300)]
    ax_hr, ax_ev = draw_timeline(fig, gs[1], beats, _summary(), None)
    assert ax_hr.figure is fig and ax_ev.figure is fig
    assert len(fig.axes) == 2
    plt.close(fig)
```

- [ ] **Step 2: Run** `.venv/bin/pytest tests/report/test_timeline.py -q` → ImportError.
- [ ] **Step 3: Implement.** Move the body of `plot_timeline` (after figure creation, before `tight_layout`/`savefig`) into

```python
def draw_timeline(fig, subplot_spec, beats, summary, start_time):
    inner = subplot_spec.subgridspec(2, 1, height_ratios=[2, 1.4], hspace=0.15)
    ax_hr = fig.add_subplot(inner[0])
    ax_ev = fig.add_subplot(inner[1], sharex=ax_hr)
    ...existing drawing...
    ax_hr.tick_params(labelbottom=False)
    return ax_hr, ax_ev
```

and make `plot_timeline` create `fig = plt.figure(figsize=(12, 5))`, call `draw_timeline(fig, GridSpec(1, 1, figure=fig)[0], ...)`, `tight_layout`, save, close.

- [ ] **Step 4: Run** `.venv/bin/pytest tests/report -q` → all pass.
- [ ] **Step 5: Commit** `git commit -am "Split timeline into a draw-into-region function"`.

### Task 2: `_draw_strip` and `_summary_lines` in `generate.py`

**Files:** Modify `src/canine_holter/report/generate.py`; Test `tests/report/test_generate.py`

- [ ] **Step 1: Failing test**

```python
def test_summary_lines_match_markdown_summary_block():
    from canine_holter.report.generate import _summary_lines
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N")]
    summary = summarize(beats)
    start = datetime(2026, 8, 23, 15, 33, 8)
    lines = _summary_lines(summary, start, duration_sec=0.8)
    assert lines[0] == "- Recording start: 2026-08-23 15:33:08"
    assert lines[1] == "- Duration: 0h 0m"
    assert "- Total beats: 2" in lines
    with tempfile.TemporaryDirectory() as out_dir:
        write_report(beats, summary, out_dir, samples=None, sample_rate=None, start_time=start)
        content = open(os.path.join(out_dir, "report.md")).read()
    for line in lines:
        assert line in content
```

- [ ] **Step 2: Run** → ImportError.
- [ ] **Step 3: Implement.** Extract the `- Recording start` … `- Sustained tachycardia events` lines into `_summary_lines(summary, start_time, duration_sec) -> list[str]`; `write_report` uses it. Extract the plotting in `_plot_strip` into `_draw_strip(ax, samples, sample_rate, center_time)`; `_plot_strip` creates the figure, calls it, sets the title, saves.
- [ ] **Step 4: Run** `.venv/bin/pytest tests/report -q` → all pass.
- [ ] **Step 5: Commit** `git commit -am "Extract summary lines and strip drawing for reuse"`.

### Task 3: `report/pdf.py`

**Files:** Create `src/canine_holter/report/pdf.py`; Test `tests/report/test_pdf.py`

- [ ] **Step 1: Failing tests**

```python
import os, re, tempfile
from datetime import datetime
import numpy as np
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import summarize
from canine_holter.report.pdf import write_pdf


def _beat(time, rr, label):
    return Beat(time=time, rr_interval=rr, qrs_duration=0.08, label=label)


def _page_count(path):
    return len(re.findall(rb"/Type\s*/Page\b(?!s)", open(path, "rb").read()))


def _beats_with_runs(n_runs):
    beats = [_beat(i * 0.8, 0.8 if i else None, "N") for i in range(200)]
    for r in range(n_runs):
        i = 20 + r * 30
        beats[i] = _beat(beats[i].time, 0.8, "V"); beats[i + 1] = _beat(beats[i + 1].time, 0.8, "V")
    return beats


def _write(n_runs, samples=True):
    beats = _beats_with_runs(n_runs)
    summary = summarize(beats)
    sig = np.sin(np.linspace(0, 2000, 160 * 100)) if samples else None
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "report.pdf")
        write_pdf(out, summary_lines=["- Total beats: 200"], beats=beats, summary=summary,
                  start_time=datetime(2026, 8, 23, 15, 33, 8),
                  samples=sig, sample_rate=100.0 if samples else None)
        return _page_count(out), os.path.getsize(out)


def test_no_flagged_events_is_one_page():
    pages, size = _write(0); assert pages == 1 and size > 1000

def test_three_events_fit_on_one_strip_page():
    assert _write(3)[0] == 2

def test_four_events_spill_to_a_third_page():
    assert _write(4)[0] == 3

def test_no_samples_lists_events_without_strip_pages():
    assert _write(3, samples=False)[0] == 1
```

- [ ] **Step 2: Run** → ModuleNotFoundError.
- [ ] **Step 3: Implement** `pdf.py`:

```python
STRIPS_PER_PAGE = 3
PAGE_SIZE_IN = (8.5, 11)

def write_pdf(out_path, *, summary_lines, beats, summary, start_time, samples, sample_rate):
    flagged = flagged_runs(beats)
    event_lines = [_event_line(i, run, start_time) for i, run in enumerate(flagged)]
    with PdfPages(out_path) as pdf:
        pdf.savefig(_summary_page(summary_lines, event_lines if samples is None else [], beats, summary, start_time)); plt.close()
        if samples is not None and sample_rate is not None:
            for page_start in range(0, len(flagged), STRIPS_PER_PAGE):
                fig = _strip_page(flagged[page_start:page_start + STRIPS_PER_PAGE], event_lines[page_start:...], samples, sample_rate)
                pdf.savefig(fig); plt.close(fig)
```

`_summary_page`: Letter figure; `fig.text` for the title (bold, 16 pt), disclaimer, and each summary line (10 pt, monospace-free, top-down with fixed line spacing); when `event_lines` is non-empty, a "Flagged events" heading and the lines; then `draw_timeline(fig, GridSpec(1, 1, figure=fig, top=0.5, bottom=0.07, left=0.09, right=0.97)[0], ...)`.
`_strip_page`: `GridSpec(STRIPS_PER_PAGE, 1, hspace=0.6, top=0.94, bottom=0.05)`; one axes per run, `_draw_strip(ax, ...)`, `ax.set_title(event_line, loc="left", fontsize=10)`.

`flagged_runs` and `_event_line` (the `Event N: k consecutive PVCs at ~label` string) move from `generate.py` into a shared spot so both markdown and PDF use them — put them in `pdf.py`'s sibling, i.e. keep them in `generate.py` but public (`flagged_runs`, `event_line`), and have `pdf.py` import from `generate.py`. To avoid a circular import, `generate.py` imports `write_pdf` lazily inside `write_report`.

- [ ] **Step 4: Run** `.venv/bin/pytest tests/report -q` → all pass.
- [ ] **Step 5: Commit** `git add -A src/canine_holter/report/pdf.py tests/report/test_pdf.py && git commit -m "Add PDF report assembler"`.

### Task 4: PDF becomes the primary artifact

**Files:** Modify `generate.py`, `pipeline.py` (docstring), `cli.py` (no code change, just prints returned path), `gui/app.py` (none); Tests: `test_generate.py`, `test_cli.py`, `test_pipeline.py`, `tests/gui/test_app.py`

- [ ] **Step 1: Failing tests.** In `test_generate.py`:

```python
def test_write_report_returns_pdf_path_and_writes_markdown_alongside():
    beats = [_beat(0.0, None, "N"), _beat(0.8, 0.8, "N")]
    summary = summarize(beats)
    with tempfile.TemporaryDirectory() as out_dir:
        path = write_report(beats, summary, out_dir, samples=None, sample_rate=None)
        assert path == os.path.join(out_dir, "report.pdf")
        assert os.path.exists(path)
        assert os.path.exists(os.path.join(out_dir, "report.md"))
```

Update existing tests: in `test_generate.py`, every `content = open(report_path).read()` / `open(path).read()` → read `os.path.join(out_dir, "report.md")`. In `test_pipeline.py` lines 31–44, 94–97, 135–143: same. In `test_cli.py`: `report_path = os.path.join(out_dir, "report.pdf")` for the printed-path assertion; existence checks on `report.md` stay (it is still written) and add the PDF where the returned path is used. In `tests/gui/test_app.py` lines 83–97: `/tmp/out/report.pdf`.

- [ ] **Step 2: Run** `.venv/bin/pytest -q` → failures in the updated tests.
- [ ] **Step 3: Implement.** End of `write_report`:

```python
    pdf_path = os.path.join(out_dir, "report.pdf")
    write_pdf(pdf_path, summary_lines=summary_lines, beats=beats, summary=summary,
              start_time=start_time, samples=samples, sample_rate=sample_rate)
    return pdf_path
```

Update docstrings in `write_report` and `run_analysis` ("Returns the path to the written PDF report; report.md and PNGs are written alongside").

- [ ] **Step 4: Run** `.venv/bin/pytest -q` → all pass.
- [ ] **Step 5: Commit** `git commit -am "Make the PDF the primary report artifact"`.

### Task 5: Verify on the real recording, docs, PR

- [ ] Run the CLI on the scratchpad `flash.dat`; open `report.pdf`, check page 1 (text + timeline) and page 2 (two strips with wall-clock titles).
- [ ] README: mention `report.pdf` as the output the CLI prints / GUI opens; CLAUDE.md `report/` bullet: add `pdf.py`.
- [ ] Commit, push, `gh pr create`.
