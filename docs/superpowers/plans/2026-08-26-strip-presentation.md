# Strip Presentation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Three-lead rhythm strips at clinical scale with beat labels, RR intervals, and plain-English captions that say what each strip shows and whether it is significant.

**Architecture:** `Recording` grows `channels`/`channel_names` (analysis lead unchanged); the DR200 and WFDB loaders fill them. `strip.py` draws N stacked panels on an ECG grid at true scale with labels and bands. `generate.py` adds a `StripCaption` per strip built from the beats and the reference bands. `pdf.py` lays out two strips per page after a primer page.

**Tech Stack:** Python 3.11, numpy, matplotlib, pytest. Spec: `docs/superpowers/specs/2026-08-26-strip-presentation-design.md`.

---

### Task 1: `Recording.channels` and the loaders

**Files:** `src/canine_holter/types.py`, `src/canine_holter/ingest/dr200.py`, `src/canine_holter/ingest/wfdb_loader.py`; tests `tests/test_types.py`, `tests/ingest/test_dr200.py`, `tests/ingest/test_wfdb_loader.py`.

- [ ] Tests: `Recording(...)` defaults `channels=None, channel_names=()`; 2-D channels of the wrong length raise `ValueError`; row count != names raises; a valid `(2, n)` with two names is accepted.
- [ ] Tests (dr200): a factory block with channel 0 codes `[7, 7, 9, 9]`, channel 1 `[1, 2, 3]`, channel 2 `[15, 14]` decodes to `channels[1][:3] == [0.0125, 0.05, 0.125]` and `channels[2][:2] == [-0.0125, -0.05]` (delta table: 1 -> +1, 2 -> +3, 3 -> +6, 15 -> -1, 14 -> -3; 12.5 uV/count); `channel_names == ("Ch 1", "Ch 2", "Ch 3")`; `load_native_flash(path, channel=1).samples` equals `channels[1]`.
- [ ] Tests (wfdb): mitdb 119 gives `channels.shape == (2, len(samples))`, `channel_names == ("MLII", "V1")`, `channels[0]` equals `samples`.
- [ ] Implement `__post_init__` validation; decode all three DR200 columns in one pass (`_DELTA_COUNTS[encoded]` is `(304, 3)`; keep a per-channel running count; zero marker rows; interpolate markers per channel); WFDB `channels=record.p_signal.T`, names `tuple(record.sig_name)`.
- [ ] `pytest tests/test_types.py tests/ingest -q`; commit "Recording carries every lead; DR200 decodes all three channels".

### Task 2: Strip drawing

**Files:** `src/canine_holter/report/strip.py`, `tests/report/test_strip.py` (rewrite).

Interface:

```python
MM_PER_SEC = 25.0
MM_PER_MV = 10.0
STRIP_WINDOW_SEC = 6.0
MARK_MARGIN_SEC = 1.0
CHANNEL_RANGE_MV = 3.0

def strip_window(center_time, mark_times, sample_rate, n_samples) -> tuple[float, float]  # (start_sec, end_sec)
def scale_label(window_sec, range_mv) -> str          # "25 mm/s · 10 mm/mV" or "21 mm/s · 7 mm/mV"
def draw_strip(fig, subplot_spec, channels, channel_names, sample_rate, center_time, beats,
               mark_times=(), pause=None, analysis_channel=0) -> list[Axes]
```

- [ ] Tests: one axes per channel (3 for a `(3, n)` array); panels share x; y-label of panel 0 ends with "(analysis)"; minor and major grid lines exist (`ax.xaxis.get_minorticklocs()` spacing 0.04, major 0.2); beat labels: texts "N" and "V" above panel 0 for beats in the window and none for beats outside; one `axvspan` patch per mark per panel; RR texts "0.80 s" and "0.42 s" printed for a marked beat and its neighbours; `pause=(30.0, 33.0)` adds an annotation whose text is "3.00 s gap"; `scale_label(6.0, 3.0) == "25 mm/s · 10 mm/mV"`, `scale_label(20.0, 3.0) == "8 mm/s · 10 mm/mV"`, `scale_label(6.0, 5.0) == "25 mm/s · 6 mm/mV"`; window clamps at the start; long marks widen the window.
- [ ] Implement per the spec section 2. Each panel plots `segment - median(segment)`, `xlim=(0, window)`, `ylim` centred on the segment midpoint with range `max(CHANNEL_RANGE_MV, ceil(ptp))`, `MultipleLocator(0.04)/(0.2)` on x and `(0.1)/(0.5)` on y, grid colours `#f6d5d5` minor / `#e8a3a3` major, trace `#1a1a1a` 0.8 lw, `axvspan(t-0.08, t+0.12, color="#c62828", alpha=0.15)` on every panel, labels via `ax.text(x, 1.05, label, transform=blended, ha="center", fontsize=8, color=..., fontweight=...)` on panel 0, RR texts at y=1.02 midway between beats (fontsize 6.5, grey), pause via `ax.annotate("", xy, xytext, arrowprops=dict(arrowstyle="<->"))` plus a centred text; tick labels only on the bottom panel every 1 s ("0 s" .. "6 s"); y tick labels off.
- [ ] `pytest tests/report/test_strip.py -q`; commit "Strips: ECG grid at clinical scale, stacked leads, beat labels, RR intervals".

### Task 3: Captions

**Files:** `src/canine_holter/report/generate.py`, `src/canine_holter/report/common.py`, `tests/report/test_generate.py`, `tests/report/test_common.py`.

- [ ] Tests: `HOW_TO_READ_STRIPS` is a non-empty list mentioning "0.2 s", "three", "provisional"; `StripSection.captions` has one entry per run in every section; isolated PVC caption `what` == "The marked beat arrived 0.40 s after the beat before it (typical here 0.80 s) and its QRS lasts 0.12 s (typical 0.08 s): early and wide is what makes it a PVC." for a `_steady(20, 0.8)` train with beat 10 replaced by `rr=0.4, qrs=0.12, label="V"`; couplet caption status "alert" and `what` naming both beats; run captions "alert" at 180 bpm and "caution" at 179; pause caption statuses at 2.49 / 2.5 / 5.0 / 5.01 s with the three significance texts; fastest/slowest captions have `status None` and the expected significance lines; caption titles use `short_time`.
- [ ] Implement `StripCaption`, `_typical_for(beats, index)` (median RR/QRS of up to 8 preceding "N" beats with measurements), `_pvc_caption`, `_run_caption`, `_extreme_captions`, and wire into `_section`/`_extremes_section`. Text pages (no samples) print title + what + significance.
- [ ] `pytest tests/report -q`; commit "Strip captions: what each strip shows and whether it is significant".

### Task 4: PDF layout

**Files:** `src/canine_holter/report/pdf.py`, `src/canine_holter/report/generate.py` (`write_report` signature), `src/canine_holter/pipeline.py`, `tests/report/test_pdf.py`, `tests/conftest.py`, `tests/test_pipeline.py`.

- [ ] Tests: `_BASE_PAGES = 4` with samples (summary, timeline, primer, one extremes page); 2 couplets -> +1 page, 3 couplets -> +2; 4 isolated PVCs -> +2; no samples -> no primer page, text pages as before; `report_text` includes "early and wide is what makes it a PVC" for the synthetic flash end-to-end run.
- [ ] Implement `STRIPS_PER_PAGE = 2`; `_primer_page()`; `_strip_page(heading, runs, captions, channels, names, sample_rate, beats)` with caption block (title bold 10 pt + scale label right-aligned, `what` wrapped at 110 chars 8.5 pt, `significance` 8.5 pt coloured by status with a "●" prefix) and a strip region 150 mm wide / (30 mm x channels) tall; `write_pdf(..., channels, channel_names, beats)`; `write_report(..., channels=None, channel_names=())` builds a one-row `ECG` array from `samples` when channels is None; pipeline passes `rec.channels`, `rec.channel_names`.
- [ ] `pytest -q`; commit "PDF: primer page, two three-lead strips per page with captions".

### Task 5: Docs and manual check

- [ ] CLAUDE.md: `Recording` contract (channels), strip description, primer page; README report paragraph.
- [ ] `canine-holter samples/teeny-2026-08-23/flash.dat --out ~/Downloads/teeny-holter-2026-08-23-v3/`; inspect pages 3-6; fix layout collisions; commit; PR.
