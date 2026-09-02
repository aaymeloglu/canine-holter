# src/canine_holter/pipeline.py
from dataclasses import replace
from datetime import date, datetime

from canine_holter.ingest.loader import load_recording
from canine_holter.detection.detect import detect_beats
from canine_holter.classify.rules import classify_beats
from canine_holter.arrhythmia.burden import summarize
from canine_holter.quality.gate import assess_quality, exclude_beats
from canine_holter.report.generate import write_report

_START_TIME_FORMATS = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%H:%M:%S", "%H:%M")


class StartTimeError(ValueError):
    """A --start-time value that matches none of the accepted forms."""


def parse_start_time(text: str, header_start: datetime | None) -> datetime:
    """Parse a user-supplied recording start time.

    Accepts ``HH:MM``, ``HH:MM:SS``, or ``YYYY-MM-DD HH:MM[:SS]``. Time-only
    forms take their date from the recording header, or today when the
    header has no clock (WFDB records). Raises StartTimeError otherwise.
    """
    for fmt in _START_TIME_FORMATS:
        try:
            parsed = datetime.strptime(text.strip(), fmt)
        except ValueError:
            continue
        if fmt.startswith("%Y"):
            return parsed
        base = header_start.date() if header_start else date.today()
        return datetime.combine(base, parsed.time())
    raise StartTimeError(
        f"unrecognized start time {text!r}; use HH:MM, HH:MM:SS, or YYYY-MM-DD HH:MM[:SS]"
    )


def run_analysis(
    input_path: str,
    out_dir: str,
    dog_weight_class: str = "medium",
    start_time: datetime | str | None = None,
) -> str:
    """Run the full ingest -> detect -> classify -> summarize -> report
    pipeline against a supported recording. Returns the path to the written
    PDF report, the only file written. WFDB records, native DR200 flash.dat recordings, and
    vendor-extracted DR200 channel files are supported.

    start_time overrides the recording's own start clock. A string is parsed
    with parse_start_time against the header (so a bare HH:MM keeps the
    recorder's date); a datetime is used as-is.

    Signal that fails quality gating (artifact, off-body, the first and
    last minute) is excluded before the detector's beats are used, a long
    off-body tail is trimmed before detection, and the report states how
    much time was analyzed and trimmed.
    """
    rec = load_recording(input_path)
    if isinstance(start_time, str):
        start_time = parse_start_time(start_time, rec.start_time)
    if start_time is not None:
        rec = replace(rec, start_time=start_time)
    quality = assess_quality(rec.samples, rec.sample_rate)
    if quality.trimmed_sec > 0:  # the off-body tail: not ECG, and detection need not chew through it
        keep = int(round(quality.duration_sec * rec.sample_rate))
        rec = replace(
            rec,
            samples=rec.samples[:keep],
            channels=None if rec.channels is None else rec.channels[:, :keep],
        )
    leads = rec.samples if rec.channels is None else rec.channels
    beats = exclude_beats(detect_beats(leads, rec.sample_rate), quality)
    labeled = classify_beats(beats)
    summary = summarize(labeled, dog_weight_class=dog_weight_class, quality=quality)
    return write_report(
        labeled,
        summary,
        out_dir,
        samples=rec.samples,
        sample_rate=rec.sample_rate,
        start_time=rec.start_time,
        channels=rec.channels,
        channel_names=rec.channel_names,
    )
