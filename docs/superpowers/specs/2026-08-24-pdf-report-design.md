# PDF Report — Design

Date: 2026-08-24

## Motivation

The report is a markdown file plus loose PNGs. Reading it means opening the
markdown in something that renders images, and the GUI's "open the report"
action lands a non-technical user in a text editor with no pictures. A
single PDF with the text, the timeline, and every strip in reading order is
what people actually want to open, print, or send to the cardiologist.

## Goals

- Write `report.pdf` alongside the existing outputs: text from the
  markdown, timeline at the top, event strips below.
- Make the PDF the primary artifact: `run_analysis` returns its path, the
  CLI prints it, the GUI opens it.
- No new dependency; the packaged `.app` must keep building unchanged.

## Non-goals

- Dropping `report.md` or the PNGs. They stay for plain-text and git-diff
  use.
- Custom fonts, a table of contents, or reproducing the markdown verbatim.
- Changing the GUI beyond which file it opens.

## Layout

US Letter, portrait (8.5 × 11 in), rendered with matplotlib's `PdfPages`.

- **Page 1:** title, disclaimer, the Summary lines exactly as they appear in
  the markdown, then the timeline (heart-rate trend + event lanes) in the
  lower half of the page.
- **Pages 2+:** flagged events, three rhythm strips per page, each titled
  with its event line (`Event 1: 2 consecutive PVCs at ~17:50:20
  (t=8232.8s)`).
- No flagged events → one page. No waveform samples (report-only mode) →
  the event lines are listed as text on page 1 and no strip pages are
  written.

## Code structure

- `report/timeline.py`: `draw_timeline(fig, subplot_spec, beats, summary,
  start_time)` draws the two-panel timeline into a region of any figure.
  `plot_timeline(...)` keeps its signature and becomes a wrapper that
  creates the 12 × 5 figure, calls `draw_timeline`, and saves the PNG.
- `report/generate.py`: `_draw_strip(ax, samples, sample_rate, center_time)`
  draws one strip into an axes; `_plot_strip` wraps it for the PNG. The
  summary lines are built by a `_summary_lines(summary, start_time,
  duration_sec)` helper so the markdown and the PDF share them.
- `report/pdf.py` (new): `write_pdf(out_path, summary_lines, event_lines,
  beats, summary, start_time, samples, sample_rate, flagged_runs)` assembles
  the pages. It knows nothing about markdown.
- `write_report` writes markdown, PNGs, and the PDF, and returns the PDF
  path. `run_analysis` passes that through; `cli.main` prints it;
  `gui.app` opens it.

## Testing

- `tests/report/test_pdf.py`: page count is 1 with no flagged events, 2 with
  three, 3 with four; a PDF is produced with `samples=None` and lists the
  events on page 1; file size is non-trivial. Page count is measured by
  counting `/Type /Page` (not `/Pages`) objects in the file bytes, which
  matplotlib writes uncompressed.
- Existing tests that read the markdown via the returned path switch to
  `os.path.join(out_dir, "report.md")`; CLI/GUI tests expect the PDF path.
- `test_timeline.py` and `test_generate.py` keep passing unchanged apart
  from the `_plot_strip` signature.
