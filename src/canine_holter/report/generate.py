"""Everything the report says, as plain data (ReportContent), built from the
labeled beats and their summary. pdf.py lays it out."""
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TypeVar
import numpy as np
from canine_holter.types import Beat, DiaryEvent
from canine_holter.arrhythmia.burden import (
    BRADYCARDIA_LINE_BPM,
    HR_EXTREME_WINDOW_BEATS,
    LONG_PAUSE_THRESHOLD_SEC,
    MIN_RUN_BEATS,
    MIN_SUCCESSIVE_DIFFERENCES,
    PAUSE_THRESHOLD_SEC,
    SUSTAINED_EVENT_MIN_BEATS,
    ArrhythmiaSummary,
    HourRow,
    RunStats,
    SinusArrest,
    escape_runs,
    pvc_runs,
    run_stats,
)
from canine_holter.classify.rules import BASELINE_WINDOW, ESCAPE_RR_RATIO
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
    pause_status,
    pvc_24h_status,
    pvc_per_24h,
    run_rate_status,
)

REPORT_TITLE = "Holter Analysis Report"
DISCLAIMER = "This is a screening aid, not a diagnosis. Review with a veterinary cardiologist."
MAX_STRIPS_PER_SECTION = 24  # 12 PDF pages at 2 strips/page; never a silent cap
EXTREMES_TITLE = "Heart-rate extremes, longest pause, fastest run"
EVENTS_TITLE = "Flagged events (couplets, triplets, VT runs)"
ISOLATED_TITLE = "Isolated PVCs"
EVENTS_SECTION_TITLE = "Diary events"
EVENT_WHAT = "The button was pressed here; the strip is centred on the press."
ESCAPE_TITLE = "Ventricular escape beats"
# One line: a strip's caption has room for two lines of "what" and one of significance.
ESCAPE_SIGNIFICANCE = "The gap is the finding (see the pauses), not the beat; several in a row need a cardiologist's look."
HOURLY_TITLE = "One strip per hour"
MAX_HOURLY_STRIPS = 48  # two days of hours; stated in the heading when it applies
HOURLY_HEADER = [
    "Hour", "Analyzed (min)", "Beats", "Min HR", "Mean HR", "Max HR", "PVCs", "Couplets",
    f"Runs ({MIN_RUN_BEATS}+)", "Escapes", "Pauses",
]
FASTEST_HR_SIGNIFICANCE = "Fast rates during play or excitement are expected; a rate this fast at rest is not."
SLOWEST_HR_SIGNIFICANCE = "Resting dogs commonly slow to this (sinus arrhythmia)."
PAUSE_SIGNIFICANCE = {
    "ok": "Under 2.5 s: within the usual range for a resting dog.",
    "caution": "Between 2.5 and 5 s: common in resting dogs with sinus arrhythmia.",
    "alert": "Over 5 s, or any pause with fainting or collapse: worth a cardiologist's review.",
}
HOW_TO_READ_TITLE = "How to read the strips that follow"
HOW_TO_READ_STRIPS = [  # under 100 characters a line: they print at 10 pt across the page
    "Each strip is a few seconds of the dog's ECG drawn on standard ECG paper: 25 mm per second and",
    "10 mm per millivolt unless the strip says otherwise. Every small square is 0.04 s; every large",
    "square (5 small) is 0.2 s. Five large squares are one second.",
    "",
    "The three rows are the same heartbeats recorded from three angles (the recorder's three",
    "channels). A beat that appears in all three rows is real; a spike in only one is usually",
    "movement or a loose electrode.",
    "",
    "Above the top row every detected beat has a label: N for a normal beat, V for a beat the",
    "software calls a PVC (premature ventricular complex), E for a ventricular escape beat (wide,",
    "but late instead of early: the ventricle filling a gap), ? when it could not measure the beat.",
    "The beats each strip is about are shaded red, and the time from the previous beat to the",
    "shaded beat, and from it to the next, is printed in seconds.",
    "",
    "A PVC is a beat with a different shape from its neighbours (usually wider) that arrives early",
    "and is usually followed by a longer gap. That is what to look for when checking a V label.",
    "One PVC on its own is common in healthy dogs; what matters is the total per 24 h (page 1).",
    "",
    "Under each strip's title: what the strip shows, with the measurements behind the label, and",
    "whether it matters, judged against the same published bands as page 1 (green / amber / red).",
    "",
    "The labels are the software's provisional calls, not a cardiologist's. The strips are here so",
    "that a reader - or the cardiologist - can check them.",
    "",
    "The last section shows one strip at the start of every hour, so the underlying rhythm can be",
    "checked through the day and night, not only where the software flagged something.",
]

T = TypeVar("T")


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
    """One rhythm strip: the beats it is about, its caption, the gap to
    bracket when it shows a pause, whether its beats are shaded (an hourly
    rhythm sample shades nothing: nothing in it is flagged), and, for a
    strip about a moment rather than beats, the moment to centre on."""
    run: list[Beat]
    caption: StripCaption
    pause: tuple[float, float] | None = None
    mark: bool = True
    center: float | None = None

    @property
    def center_time(self) -> float:
        if self.center is not None:
            return self.center
        return (self.run[0].time + self.run[-1].time) / 2


@dataclass(frozen=True)
class StripSection:
    heading: str
    items: list[StripItem]


@dataclass(frozen=True)
class ReportContent:
    """Everything textual in the report, independent of how it is rendered."""
    title: str
    disclaimer: str
    summary_groups: list[SummaryGroup]
    footer_lines: list[str]
    primer_title: str
    primer_lines: list[str]
    sections: list[StripSection]
    hourly_header: list[str]
    hourly_rows: list[list[str]]  # one row of cells per HourRow, in HOURLY_HEADER order


def format_duration(duration_sec: float) -> str:
    hours, rem = divmod(int(duration_sec), 3600)
    return f"{hours}h {rem // 60}m"


def short_time(elapsed_sec: float, start_time: datetime | None) -> str:
    """Clock time when the recording start is known, else elapsed seconds."""
    if start_time is None:
        return f"t={elapsed_sec:.0f}s"
    return (start_time + timedelta(seconds=elapsed_sec)).strftime("%H:%M:%S")


def select_evenly(items: list[T], max_n: int) -> list[T]:
    """All of items if there are at most max_n, else max_n of them spread
    evenly from first to last: a fair sample of a long recording rather
    than its first few minutes."""
    if len(items) <= max_n:
        return list(items)
    step = (len(items) - 1) / (max_n - 1)
    return [items[round(i * step)] for i in range(max_n)]


def _sec(value: float | None) -> str:
    return f"{value:.2f} s" if value is not None else "n/a"


# --- summary panels -----------------------------------------------------------


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
            str(row.escapes),
            str(row.pauses),
        ]
        for row in summary.hourly
    ]


def _recorder_ran_rows(summary: ArrhythmiaSummary) -> list[SummaryRow]:
    """Only when an off-body tail was trimmed: the panel's reference column
    is narrow, so the recorder's full run time gets its own row."""
    if summary.trimmed_sec <= 0:
        return []
    ran = format_duration(summary.duration_sec + summary.trimmed_sec)
    return [SummaryRow("Recorder ran", ran, "off-body tail trimmed")]


def _recording_group(
    summary: ArrhythmiaSummary, start_time: datetime | None, events: tuple[DiaryEvent, ...] = ()
) -> SummaryGroup:
    duration, analyzed = summary.duration_sec, summary.analyzed_sec
    pct = 100.0 * analyzed / duration if duration else 0.0
    return SummaryGroup("Recording", [
        SummaryRow("Start", start_time.strftime("%Y-%m-%d %H:%M:%S") if start_time else "unknown"),
        SummaryRow("Duration", format_duration(duration)),
        *_recorder_ran_rows(summary),
        SummaryRow("Analyzed", f"{format_duration(analyzed)} ({pct:.0f}%)", ANALYZED_BAND, analyzed_status(analyzed)),
        SummaryRow("Excluded", format_duration(duration - analyzed), "artifact / off-body"),
        SummaryRow("Total beats", str(summary.total_beats)),
        SummaryRow("Diary events", str(len(events)), "button presses"),
    ])


def _share(count: int, total: int) -> str:
    return f"{count} ({100.0 * count / total:.0f}%)" if total else "0 (0%)"


def _heart_rate_group(summary: ArrhythmiaSummary, start_time: datetime | None) -> SummaryGroup:
    hr = summary.heart_rate
    window = f"{HR_EXTREME_WINDOW_BEATS}-beat median"
    brady, tachy = summary.brady_threshold_bpm, summary.tachy_threshold_bpm
    if hr is None:
        rate_rows = [
            SummaryRow("Heart rate", f"not computed (fewer than {HR_EXTREME_WINDOW_BEATS} beats with an RR)")
        ]
    else:
        rate_rows = [
            SummaryRow("Mean", f"{hr.mean_bpm:.0f} bpm"),
            SummaryRow("Slowest", f"{hr.min_bpm:.0f} bpm at {short_time(hr.min_time, start_time)}", window),
            SummaryRow("Fastest", f"{hr.max_bpm:.0f} bpm at {short_time(hr.max_time, start_time)}", window),
            SummaryRow(f"Under {BRADYCARDIA_LINE_BPM:g} bpm", _share(summary.slow_beats_at_line, summary.rated_beats), window),
        ] + (
            # The cardiologist reads against 60 bpm; the class threshold is printed beside it unless it is the same line.
            [SummaryRow(f"Under {brady:g} bpm", _share(summary.slow_beats, summary.rated_beats), window)]
            if brady != BRADYCARDIA_LINE_BPM else []
        ) + [
            SummaryRow(f"Over {tachy:g} bpm", _share(summary.fast_beats, summary.rated_beats), window),
        ]
    return SummaryGroup("Heart rate", rate_rows + [
        SummaryRow(
            "Brady events", str(len(summary.bradycardia_events)), f"{SUSTAINED_EVENT_MIN_BEATS}+ beats < {brady:g} bpm"
        ),
        SummaryRow(
            "Tachy events", str(len(summary.tachycardia_events)), f"{SUSTAINED_EVENT_MIN_BEATS}+ beats > {tachy:g} bpm"
        ),
    ])


def _supraventricular_group() -> SummaryGroup:
    """Stated, not counted: a premature narrow beat cannot be told from
    sinus arrhythmia by timing alone (pNN50 runs ~70% in a resting dog),
    and P waves are not analyzed. Absent must read as absent, not zero."""
    return SummaryGroup("Supraventricular ectopy", [SummaryRow("SVPBs", "not assessed", "needs P-wave analysis")])


def _run_text(run: RunStats, start_time: datetime | None) -> str:
    """Beats and rate only: the time is on the run's strip, and the panel
    row has no room for it beside the rate band."""
    return f"{run.beats} beats, {run.bpm:.0f} bpm"


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
        SummaryRow("Escape beats", str(len(summary.escape_beats)), f"wide, RR >= {ESCAPE_RR_RATIO:g}x local"),
        SummaryRow("Escape couplets", str(summary.escape_couplets)),
        SummaryRow(f"Escape runs ({MIN_RUN_BEATS}+)", str(summary.escape_runs)),
        SummaryRow("Longest run", _run_text(longest, start_time) if longest else "none"),
        SummaryRow(
            "Fastest run",
            _run_text(fastest, start_time) if fastest else "none",
            RUN_RATE_BAND,
            run_rate_status(fastest.bpm if fastest else None),
        ),
    ])


def _escape_count(n: int, kind: str = "escape beat") -> str:
    return f"{n} {kind}{'s' if n != 1 else ''}"


def _sinus_interval_row(interval: SinusArrest | None) -> SummaryRow:
    """The longest gap between sinus beats: an escape beat interrupts a
    sinus arrest without ending it, so this can exceed the longest RR."""
    if interval is None:
        return SummaryRow("Sinus interval", "n/a", "longest")
    reference = f"longest; {_escape_count(interval.escape_beats)} inside" if interval.escape_beats else "longest: the longest pause"
    return SummaryRow("Sinus interval", f"{interval.duration_sec:.2f} s", reference, pause_status(interval.duration_sec))


def _pause_group(summary: ArrhythmiaSummary) -> SummaryGroup:
    longest = summary.longest_pause_sec
    return SummaryGroup("Pauses", [
        SummaryRow("Pauses", str(len(summary.pauses)), f">= {PAUSE_THRESHOLD_SEC:g} s"),
        SummaryRow(f"Pauses > {LONG_PAUSE_THRESHOLD_SEC:g} s", str(summary.long_pauses)),
        SummaryRow(
            "Longest", f"{longest:.2f} s" if longest is not None else "n/a", PAUSE_BAND, pause_status(longest)
        ),
        _sinus_interval_row(summary.longest_sinus_interval),
        SummaryRow("Sinus arrests", str(len(summary.sinus_arrests)), "bridged by escape beats"),
    ])


def _variability_group(summary: ArrhythmiaSummary) -> SummaryGroup:
    """Uncoloured: there are no canine reference bands for these."""
    hrv = summary.heart_rate_variability
    if hrv is None:
        rows = [SummaryRow(
            "RR variability", f"not computed (fewer than {MIN_SUCCESSIVE_DIFFERENCES} successive NN differences)"
        )]
    else:
        rows = [
            SummaryRow("SDNN", f"{hrv.sdnn_ms:.0f} ms", f"{hrv.nn_intervals} NN intervals"),
            SummaryRow("RMSSD", f"{hrv.rmssd_ms:.0f} ms"),
            SummaryRow("pNN50", f"{hrv.pnn50_pct:.0f}%"),
        ]
    return SummaryGroup("RR variability", rows)


def summary_groups(
    summary: ArrhythmiaSummary, start_time: datetime | None, events: tuple[DiaryEvent, ...] = ()
) -> list[SummaryGroup]:
    """The six summary panels, in reading order (a 3x2 grid on the page)."""
    return [
        _recording_group(summary, start_time, events),
        _heart_rate_group(summary, start_time),
        _ectopy_group(summary, start_time),
        _supraventricular_group(),
        _pause_group(summary),
        _variability_group(summary),
    ]


# --- strips -------------------------------------------------------------------


def _typical(beats: list[Beat], index: int) -> tuple[float | None, float | None]:
    """Median RR and QRS of the up-to-BASELINE_WINDOW normal beats before
    index: the same baseline the classifier compared the beat with."""
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
        # A lone PVC is common; the primer says so once rather than every strip.
        return StripCaption(f"Isolated PVC {index + 1} · {when}", what, "")
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


def _capped_heading(title: str, shown: int, total: int) -> str:
    return title if shown == total else f"{title} ({shown} of {total} shown, evenly spaced through the recording)"


def _section(title: str, runs: list[list[Beat]], beats: list[Beat], start_time: datetime | None) -> StripSection | None:
    """A capped section of PVC runs, with the cap spelled out in the heading
    whenever it applied."""
    if not runs:
        return None
    shown = select_evenly(runs, MAX_STRIPS_PER_SECTION)
    return StripSection(_capped_heading(title, len(shown), len(runs)), [
        StripItem(run, _pvc_caption(i, run, beats, start_time)) for i, run in enumerate(shown)
    ])


def _escape_caption(index: int, run: list[Beat], beats: list[Beat], start_time: datetime | None) -> StripCaption:
    """Caption for one escape beat, or for a run of them (a slow couplet, or
    an idioventricular rhythm at three or more)."""
    when = short_time(run[0].time, start_time)
    if len(run) == 1:
        beat = run[0]
        typical_rr, typical_qrs = _typical(beats, next(i for i, b in enumerate(beats) if b.time == beat.time))
        what = (
            f"The marked beat arrived {_sec(beat.rr_interval)} after the beat before it (typical here"
            f" {_sec(typical_rr)}) and its QRS lasts {_sec(beat.qrs_duration)} (typical {_sec(typical_qrs)}):"
            " wide and late is what makes it a ventricular escape beat."
        )
        return StripCaption(f"Escape beat {index + 1} · {when}", what, ESCAPE_SIGNIFICANCE)
    rrs = [b.rr_interval for b in run[1:] if b.rr_interval]
    rate = f" at {60.0 / float(np.mean(rrs)):.0f} bpm" if rrs else ""
    if len(run) == 2:
        return StripCaption(
            f"Escape couplet {index + 1} · {when}",
            f"2 escape beats in a row{rate}: a slow couplet.",
            ESCAPE_SIGNIFICANCE,
        )
    return StripCaption(
        f"Escape run {index + 1} · {when}",
        f"{len(run)} escape beats in a row{rate}: an idioventricular rhythm.",
        ESCAPE_SIGNIFICANCE,
    )


def _escape_section(beats: list[Beat], start_time: datetime | None) -> StripSection | None:
    """Every escape beat or run of them gets a strip, capped like the PVC
    sections: the width call behind an E is as reviewable, and as fallible,
    as a V."""
    runs = escape_runs(beats)
    if not runs:
        return None
    shown = select_evenly(runs, MAX_STRIPS_PER_SECTION)
    return StripSection(_capped_heading(ESCAPE_TITLE, len(shown), len(runs)), [
        StripItem(run, _escape_caption(i, run, beats, start_time)) for i, run in enumerate(shown)
    ])


def _events_section(events: tuple[DiaryEvent, ...], start_time: datetime | None) -> StripSection | None:
    """One strip per diary-button press, centred on the press: when the
    dog has collapse episodes, this is the strip a cardiologist wants
    first. Nothing is shaded; the leads show whatever was recorded, even
    inside an excluded span."""
    if not events:
        return None
    shown = select_evenly(list(events), MAX_STRIPS_PER_SECTION)
    return StripSection(_capped_heading(EVENTS_SECTION_TITLE, len(shown), len(events)), [
        StripItem(
            [], StripCaption(f"{event.label} · {short_time(event.time_sec, start_time)}", EVENT_WHAT, ""),
            mark=False, center=event.time_sec,
        )
        for event in shown
    ])


def _hour_rate_text(row: HourRow) -> str:
    if row.mean_bpm is None:
        return "Too few beats this hour for a rate."
    return f"This hour: {row.min_bpm:.0f}-{row.max_bpm:.0f} bpm, mean {row.mean_bpm:.0f}."


def _hourly_section(beats: list[Beat], summary: ArrhythmiaSummary, start_time: datetime | None) -> StripSection | None:
    """A strip at the first beat of every hour that has one: the underlying
    rhythm, checkable through the day and not only where something was
    flagged (HE/LX prints the same "one per hour" strips)."""
    rows = [row for row in summary.hourly if row.beats]
    if not rows:
        return None
    shown = select_evenly(rows, MAX_HOURLY_STRIPS)
    items = []
    for row in shown:
        beat = next(b for b in beats if b.time >= row.start_sec)
        items.append(StripItem([beat], StripCaption(
            f"Hour {_hour_label(row, start_time)} · {short_time(beat.time, start_time)}",
            f"The first beats of the hour. {_hour_rate_text(row)}",
            "",
        ), mark=False))
    return StripSection(_capped_heading(HOURLY_TITLE, len(shown), len(rows)), items)


def _beat_at(beats: list[Beat], time: float) -> Beat:
    return next(b for b in beats if b.time == time)


def _pause_beats(beats: list[Beat], summary: ArrhythmiaSummary) -> list[Beat]:
    """The two beats bracketing the longest pause, so the strip is centred on
    the gap and marks both edges."""
    end = next(i for i, b in enumerate(beats) if b.rr_interval == summary.longest_pause_sec)
    return beats[max(0, end - 1) : end + 1]


def _extremes_section(beats: list[Beat], summary: ArrhythmiaSummary, start_time: datetime | None) -> StripSection | None:
    """Strips a reader looks at first: fastest and slowest heart rate, the
    longest pause (when one crossed the threshold), the longest sinus
    interval when an escape beat bridged it (otherwise it is the longest
    pause), and the fastest run (when there is one). Absent without heart-rate stats, since a recording that
    short has nothing to show."""
    hr = summary.heart_rate
    if hr is None:
        return None
    window = f"{HR_EXTREME_WINDOW_BEATS} beats"
    items = [
        StripItem([_beat_at(beats, hr.max_time)], StripCaption(
            f"Fastest heart rate · {short_time(hr.max_time, start_time)}",
            f"{hr.max_bpm:.0f} bpm averaged over {window}.",
            FASTEST_HR_SIGNIFICANCE,
        )),
        StripItem([_beat_at(beats, hr.min_time)], StripCaption(
            f"Slowest heart rate · {short_time(hr.min_time, start_time)}",
            f"{hr.min_bpm:.0f} bpm averaged over {window}; the gaps between beats are printed in seconds.",
            SLOWEST_HR_SIGNIFICANCE,
        )),
    ]
    if summary.pauses:
        pause = _pause_beats(beats, summary)
        status = pause_status(summary.longest_pause_sec)
        items.append(StripItem(
            pause,
            StripCaption(
                f"Longest pause · {short_time(pause[-1].time, start_time)}",
                f"No beat for {summary.longest_pause_sec:.2f} s.",
                PAUSE_SIGNIFICANCE[status],
                status,
            ),
            pause=(pause[0].time, pause[-1].time),
        ))
    arrest = summary.longest_sinus_interval
    if arrest is not None and arrest.escape_beats:
        status = pause_status(arrest.duration_sec)
        items.append(StripItem(
            [b for b in beats if arrest.start_time < b.time < arrest.end_time],
            StripCaption(
                f"Longest sinus interval · {short_time(arrest.end_time, start_time)}",
                f"No sinus beat for {arrest.duration_sec:.2f} s;"
                f" {_escape_count(arrest.escape_beats, 'ventricular escape beat')} filled the gap.",
                PAUSE_SIGNIFICANCE[status],
                status,
            ),
            pause=(arrest.start_time, arrest.end_time),
        ))
    if summary.fastest_run is not None:
        run = next(r for r in pvc_runs(beats) if r[0].time == summary.fastest_run.start_time)
        caption = _pvc_caption(0, run, beats, start_time)
        items.append(StripItem(run, StripCaption(
            f"Fastest run · {short_time(run[0].time, start_time)}",
            caption.what,
            caption.significance,
            caption.status,
        )))
    return StripSection(EXTREMES_TITLE, items)


def build_content(
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    start_time: datetime | None,
    events: tuple[DiaryEvent, ...] = (),
) -> ReportContent:
    """Assemble the report content: summary panels, the strip sections
    (heart-rate extremes, then the diary-button presses, then flagged
    multi-beat runs, then isolated PVCs, then ventricular escape beats,
    then one strip per hour; the capped sections state their cap in the
    heading), and the hourly table. Event times are wall-clock labels
    when start_time is known."""
    runs = pvc_runs(beats)
    sections = [
        _extremes_section(beats, summary, start_time),
        _events_section(events, start_time),
        _section(EVENTS_TITLE, [r for r in runs if len(r) >= 2], beats, start_time),
        _section(ISOLATED_TITLE, [r for r in runs if len(r) == 1], beats, start_time),
        _escape_section(beats, start_time),
        _hourly_section(beats, summary, start_time),
    ]
    return ReportContent(
        title=REPORT_TITLE,
        disclaimer=DISCLAIMER,
        summary_groups=summary_groups(summary, start_time, events),
        footer_lines=list(FOOTER_LINES),
        primer_title=HOW_TO_READ_TITLE,
        primer_lines=HOW_TO_READ_STRIPS,
        sections=[s for s in sections if s is not None],
        hourly_header=HOURLY_HEADER,
        hourly_rows=_hourly_rows(summary, start_time),
    )
