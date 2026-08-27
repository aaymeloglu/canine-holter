"""Turns labeled beats + summary into the report content (plain data) and
writes it as report.pdf - the only output file."""
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
import numpy as np
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import (
    HR_EXTREME_WINDOW_BEATS,
    MIN_RUN_BEATS,
    PAUSE_THRESHOLD_SEC,
    ArrhythmiaSummary,
    HourRow,
    RunStats,
    pvc_runs,
    run_stats,
)
from canine_holter.classify.rules import BASELINE_WINDOW
from canine_holter.report.common import (
    EVENTS_TITLE,
    EXTREMES_TITLE,
    ISOLATED_TITLE,
    MAX_STRIPS_PER_SECTION,
    event_line,
    flagged_runs,
    format_time,
    isolated_pvcs,
    pvc_line,
    section_heading,
    select_evenly,
    short_time,
)
from canine_holter.report.pdf import write_pdf
from canine_holter.report.reference import (
    ANALYZED_BAND,
    COUNT_BAND,
    FOOTER_LINES,
    MIN_HOURS_FOR_24H_SCALING,
    PAUSE_BAND,
    PVC_24H_BAND,
    RUN_RATE_BAND,
    analyzed_status,
    count_status,
    format_duration,
    pause_status,
    pvc_24h_status,
    pvc_per_24h,
    run_rate_status,
)


@dataclass(frozen=True)
class SummaryRow:
    """One line of a summary panel: the value, the published band it is
    compared with (short text beside it), and where it falls."""
    label: str
    value: str
    reference: str = ""
    status: str | None = None  # "ok" | "caution" | "alert" | None (uncolored)


@dataclass(frozen=True)
class SummaryGroup:
    title: str
    rows: list[SummaryRow]


@dataclass(frozen=True)
class StripCaption:
    """The words under a strip's title: what it shows (with the measured
    numbers behind the software's call) and whether it is significant,
    with the status that colours the significance line."""
    title: str
    what: str
    significance: str
    status: str | None = None


@dataclass(frozen=True)
class StripItem:
    """One rhythm strip and the text/annotation rendered with it."""
    run: list[Beat]
    label: str
    caption: StripCaption
    pause: tuple[float, float] | None = None


@dataclass(frozen=True)
class StripSection:
    """One capped section of rhythm strips."""
    heading: str
    items: list[StripItem]


@dataclass(frozen=True)
class ReportContent:
    """Everything textual in the report, independent of how it is rendered."""
    summary_groups: list[SummaryGroup]
    footer_lines: list[str]
    sections: list[StripSection]
    hourly_header: list[str]
    hourly_rows: list[list[str]]  # one row of cells per HourRow, in HOURLY_HEADER order


HOURLY_HEADER = [
    "Hour", "Analyzed (min)", "Beats", "Min HR", "Mean HR", "Max HR", "PVCs", "Couplets",
    f"Runs ({MIN_RUN_BEATS}+)", "Pauses",
]


def _hour_label(row: HourRow, start_time: datetime | None) -> str:
    """'15:33-16:33' with a known start, else elapsed 'h:mm-h:mm'."""
    def clock(sec: float) -> str:
        if start_time is not None:
            return (start_time + timedelta(seconds=sec)).strftime("%H:%M")
        return f"{int(sec // 3600)}:{int(sec % 3600 // 60):02d}"
    return f"{clock(row.start_sec)}-{clock(row.end_sec)}"


def _hourly_rows(summary: ArrhythmiaSummary, start_time: datetime | None) -> list[list[str]]:
    def bpm(value: float | None) -> str:
        return "-" if value is None else f"{value:.0f}"
    return [
        [
            _hour_label(row, start_time),
            f"{row.analyzed_sec / 60:.1f}",
            str(row.beats),
            bpm(row.min_bpm),
            bpm(row.mean_bpm),
            bpm(row.max_bpm),
            str(row.pvcs),
            str(row.couplets),
            str(row.runs),
            str(row.pauses),
        ]
        for row in summary.hourly
    ]


def _recording_group(summary: ArrhythmiaSummary, start_time: datetime | None) -> SummaryGroup:
    duration, analyzed = summary.duration_sec, summary.analyzed_sec
    pct = 100.0 * analyzed / duration if duration else 0.0
    return SummaryGroup("Recording", [
        SummaryRow("Start", start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else "unknown"),
        SummaryRow("Duration", format_duration(duration)),
        SummaryRow("Analyzed", f"{format_duration(analyzed)} ({pct:.0f}%)", ANALYZED_BAND, analyzed_status(analyzed)),
        SummaryRow("Excluded", format_duration(duration - analyzed), "artifact / off-body"),
        SummaryRow("Total beats", str(summary.total_beats)),
    ])


def _heart_rate_group(summary: ArrhythmiaSummary, start_time: datetime | None) -> SummaryGroup:
    hr = summary.heart_rate
    window = f"{HR_EXTREME_WINDOW_BEATS}-beat median"
    if hr is None:
        rate_rows = [
            SummaryRow("Heart rate", f"not computed (fewer than {HR_EXTREME_WINDOW_BEATS} beats with an RR)")
        ]
    else:
        rate_rows = [
            SummaryRow("Mean", f"{hr.mean_bpm:.0f} bpm"),
            SummaryRow("Slowest", f"{hr.min_bpm:.0f} bpm at {short_time(hr.min_time, start_time)}", window),
            SummaryRow("Fastest", f"{hr.max_bpm:.0f} bpm at {short_time(hr.max_time, start_time)}", window),
        ]
    return SummaryGroup("Heart rate", rate_rows + [
        SummaryRow("Brady events", str(len(summary.bradycardia_events))),
        SummaryRow("Tachy events", str(len(summary.tachycardia_events))),
    ])


def _run_text(run: RunStats, start_time: datetime | None) -> str:
    return f"{run.beats} beats, {run.bpm:.0f} bpm, {short_time(run.start_time, start_time)}"


def _pvc_24h_row(summary: ArrhythmiaSummary) -> SummaryRow:
    scaled = pvc_per_24h(summary.pvc_count, summary.analyzed_sec)
    if scaled is None:
        return SummaryRow("PVCs per 24 h", "n/a", f"needs >= {MIN_HOURS_FOR_24H_SCALING} h analyzed")
    return SummaryRow("PVCs per 24 h", str(round(scaled)), PVC_24H_BAND, pvc_24h_status(scaled))


def _ectopy_group(summary: ArrhythmiaSummary, start_time: datetime | None) -> SummaryGroup:
    longest, fastest = summary.longest_run, summary.fastest_run
    return SummaryGroup("Ventricular ectopy", [
        SummaryRow("PVCs", f"{summary.pvc_count} ({summary.pvc_burden_pct:.2f}%)"),
        _pvc_24h_row(summary),
        SummaryRow("Couplets", str(summary.couplets), COUNT_BAND, count_status(summary.couplets)),
        SummaryRow("Triplets", str(summary.triplets), COUNT_BAND, count_status(summary.triplets)),
        SummaryRow("VT runs (4+)", str(summary.vtach_runs), COUNT_BAND, count_status(summary.vtach_runs)),
        SummaryRow("Longest run", _run_text(longest, start_time) if longest else "none"),
        SummaryRow(
            "Fastest run",
            _run_text(fastest, start_time) if fastest else "none",
            RUN_RATE_BAND,
            run_rate_status(fastest.bpm if fastest else None),
        ),
    ])


def _pause_group(summary: ArrhythmiaSummary) -> SummaryGroup:
    longest = summary.longest_pause_sec
    return SummaryGroup("Pauses", [
        SummaryRow("Pauses", str(len(summary.pauses)), f">= {PAUSE_THRESHOLD_SEC:g} s"),
        SummaryRow(
            "Longest", f"{longest:.2f} s" if longest is not None else "n/a", PAUSE_BAND, pause_status(longest)
        ),
    ])


def summary_groups(summary: ArrhythmiaSummary, start_time: datetime | None) -> list[SummaryGroup]:
    """The four summary panels, in reading order."""
    return [
        _recording_group(summary, start_time),
        _heart_rate_group(summary, start_time),
        _ectopy_group(summary, start_time),
        _pause_group(summary),
    ]


ISOLATED_SIGNIFICANCE = ""  # said once in the primer (common.HOW_TO_READ_STRIPS), not under every strip
FASTEST_HR_SIGNIFICANCE = "Fast rates during play or excitement are expected; a rate this fast at rest is not."
SLOWEST_HR_SIGNIFICANCE = "Resting dogs commonly slow to this (sinus arrhythmia)."
PAUSE_SIGNIFICANCE = {
    "ok": "Under 2.5 s: within the usual range for a resting dog.",
    "caution": "Between 2.5 and 5 s: common in resting dogs with sinus arrhythmia.",
    "alert": "Over 5 s, or any pause with fainting or collapse: worth a cardiologist's review.",
}


def _sec(value: float | None) -> str:
    return f"{value:.2f} s" if value is not None else "n/a"


def _typical(beats: list[Beat], index: int) -> tuple[float | None, float | None]:
    """Median RR and QRS of the up-to-BASELINE_WINDOW normal beats before
    index - the same baseline the classifier compared the beat with."""
    previous = [
        b for b in beats[max(0, index - 5 * BASELINE_WINDOW) : index]
        if b.label == "N" and b.rr_interval and b.qrs_duration
    ][-BASELINE_WINDOW:]
    if not previous:
        return None, None
    return (
        float(np.median([b.rr_interval for b in previous])),
        float(np.median([b.qrs_duration for b in previous])),
    )


def _pvc_caption(index: int, run: list[Beat], beats: list[Beat], start_time: datetime | None) -> StripCaption:
    """Caption for an isolated PVC, a couplet, a triplet, or a longer run."""
    first_index = next(i for i, b in enumerate(beats) if b.time == run[0].time)
    typical_rr, typical_qrs = _typical(beats, first_index)
    n = len(run)
    when = short_time(run[0].time, start_time)  # the first beat, as RunStats.start_time on page 1
    if n == 1:
        beat = run[0]
        what = (
            f"The marked beat arrived {_sec(beat.rr_interval)} after the beat before it (typical here"
            f" {_sec(typical_rr)}) and its QRS lasts {_sec(beat.qrs_duration)} (typical {_sec(typical_qrs)}):"
            " early and wide is what makes it a PVC."
        )
        return StripCaption(f"Isolated PVC {index + 1} · {when}", what, ISOLATED_SIGNIFICANCE)
    if n < 4:
        rrs = " and ".join(_sec(b.rr_interval) for b in run)
        qrss = " and ".join(_sec(b.qrs_duration) for b in run)
        what = (
            f"The marked beats arrived {rrs} after the beat before them (typical here {_sec(typical_rr)}),"
            f" with QRS of {qrss} (typical {_sec(typical_qrs)}): early and wide is what makes them PVCs."
        )
        kind = "couplet" if n == 2 else "triplet"
        significance = f"Any {kind} is worth a cardiologist's review, whatever the PVC count."
        return StripCaption(f"Event {index + 1} · {when}", what, significance, "alert")
    stats = next((r for r in run_stats(beats) if r.start_time == run[0].time), None)
    what = (
        f"{n} beats in a row, each early and wide; the first arrived {_sec(run[0].rr_interval)} after the"
        f" beat before it (typical here {_sec(typical_rr)})."
    )
    significance, status = _run_significance(n, stats.bpm if stats else None)
    return StripCaption(f"Event {index + 1} · {when}", what, significance, status)


def _run_significance(n: int, bpm: float | None) -> tuple[str, str]:
    if bpm is None:
        return f"{n} PVCs in a row; the rate could not be measured.", "caution"
    if run_rate_status(bpm) == "alert":
        return f"{n} PVCs in a row at {bpm:.0f} bpm is ventricular tachycardia.", "alert"
    return (
        f"{n} PVCs in a row at {bpm:.0f} bpm: an accelerated idioventricular rhythm, generally less"
        " concerning than ventricular tachycardia.",
        "caution",
    )


def _section(title: str, runs: list[list[Beat]], labeler, beats: list[Beat], start_time: datetime | None) -> StripSection | None:
    if not runs:
        return None
    shown = select_evenly(runs, MAX_STRIPS_PER_SECTION)
    items = [
        StripItem(
            run=run,
            label=labeler(i, run, start_time),
            caption=_pvc_caption(i, run, beats, start_time),
        )
        for i, run in enumerate(shown)
    ]
    return StripSection(section_heading(title, len(shown), len(runs)), items)


def _beat_at(beats: list[Beat], time: float) -> Beat:
    return next(b for b in beats if b.time == time)


def _pause_beats(beats: list[Beat], summary: ArrhythmiaSummary) -> list[Beat]:
    """The two beats bracketing the longest pause, so the strip is centred on
    the gap and marks both edges."""
    end = next(i for i, b in enumerate(beats) if b.rr_interval == summary.longest_pause_sec)
    return beats[max(0, end - 1) : end + 1]


def _extremes_section(beats: list[Beat], summary: ArrhythmiaSummary, start_time: datetime | None) -> StripSection | None:
    """Strips a reader looks at first: fastest and slowest heart rate, the
    longest pause (when one crossed the threshold), and the fastest run (when
    there is one). Absent without heart-rate stats, since a recording that
    short has nothing to show."""
    hr = summary.heart_rate
    if hr is None:
        return None
    window = f"{HR_EXTREME_WINDOW_BEATS} beats"
    items = [
        StripItem(
            run=[_beat_at(beats, hr.max_time)],
            label=f"Fastest heart rate: {hr.max_bpm:.0f} bpm at {format_time(hr.max_time, start_time)}",
            caption=StripCaption(
                f"Fastest heart rate · {short_time(hr.max_time, start_time)}",
                f"{hr.max_bpm:.0f} bpm averaged over {window}.",
                FASTEST_HR_SIGNIFICANCE,
            ),
        ),
        StripItem(
            run=[_beat_at(beats, hr.min_time)],
            label=f"Slowest heart rate: {hr.min_bpm:.0f} bpm at {format_time(hr.min_time, start_time)}",
            caption=StripCaption(
                f"Slowest heart rate · {short_time(hr.min_time, start_time)}",
                f"{hr.min_bpm:.0f} bpm averaged over {window}; the gaps between beats are printed in seconds.",
                SLOWEST_HR_SIGNIFICANCE,
            ),
        ),
    ]
    if summary.pauses:
        pause = _pause_beats(beats, summary)
        status = pause_status(summary.longest_pause_sec)
        items.append(StripItem(
            run=pause,
            label=(
                f"Longest pause: {summary.longest_pause_sec:.2f} s, ending at "
                f"{format_time(pause[-1].time, start_time)}"
            ),
            caption=StripCaption(
                f"Longest pause · {short_time(pause[-1].time, start_time)}",
                f"No beat for {summary.longest_pause_sec:.2f} s.",
                PAUSE_SIGNIFICANCE[status],
                status,
            ),
            pause=(pause[0].time, pause[-1].time),
        ))
    if summary.fastest_run is not None:
        run = next(r for r in pvc_runs(beats) if r[0].time == summary.fastest_run.start_time)
        caption = _pvc_caption(0, run, beats, start_time)
        items.append(StripItem(
            run=run,
            label=f"Fastest run: {_run_text(summary.fastest_run, start_time)}",
            caption=StripCaption(
                f"Fastest run · {short_time(run[0].time, start_time)}",
                caption.what,
                caption.significance,
                caption.status,
            ),
        ))
    return StripSection(EXTREMES_TITLE, items)


def build_content(beats: list[Beat], summary: ArrhythmiaSummary, start_time: datetime | None) -> ReportContent:
    """Assemble the report content: summary panels, the strip sections
    (heart-rate extremes, then flagged multi-beat runs, then isolated PVCs;
    the latter two capped at MAX_STRIPS_PER_SECTION with the cap stated in
    the heading), and the hourly table. Event times are wall-clock labels
    when start_time is known."""
    sections = [
        _extremes_section(beats, summary, start_time),
        _section(EVENTS_TITLE, flagged_runs(beats), event_line, beats, start_time),
        _section(ISOLATED_TITLE, isolated_pvcs(beats), pvc_line, beats, start_time),
    ]
    return ReportContent(
        summary_groups=summary_groups(summary, start_time),
        footer_lines=list(FOOTER_LINES),
        sections=[s for s in sections if s is not None],
        hourly_header=HOURLY_HEADER,
        hourly_rows=_hourly_rows(summary, start_time),
    )


def write_report(
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    out_dir: str,
    samples: np.ndarray | None,
    sample_rate: float | None,
    start_time: datetime | None = None,
    channels: np.ndarray | None = None,
    channel_names: tuple[str, ...] = (),
) -> str:
    """Write report.pdf into out_dir and return its path. Nothing else is
    written. Strips show every lead in channels, or the analysis lead alone
    when only samples are given; without any waveform they are listed as
    text."""
    os.makedirs(out_dir, exist_ok=True)
    pdf_path = os.path.join(out_dir, "report.pdf")
    if channels is None and samples is not None:
        channels, channel_names = samples[None, :], ("ECG",)
    write_pdf(
        pdf_path,
        content=build_content(beats, summary, start_time),
        beats=beats,
        summary=summary,
        start_time=start_time,
        channels=channels,
        channel_names=channel_names,
        sample_rate=sample_rate,
    )
    return pdf_path
