"""Decide which stretches of a recording are analyzable ECG and which are
artifact (off-body, lead-off, saturation, flat line, hookup and removal),
so nothing downstream counts a beat, pause, or run inside them.

The rules are amplitude, flat-line, and lead-off tone, judged per window;
the amplitude rules compare against the recording's own median (DR200
samples carry a decoder DC offset and gain varies by recorder and lead),
taken over the windows the two absolute rules did not condemn, so a
recording that is mostly off-body still measures its ECG. Kurtosis- and spectrum-based noise measures
were tested and rejected: they exclude ventricular flutter and VT, which
are near-sinusoidal like noise, and those are exactly what this tool must
keep. Evidence and rejected rules:
docs/superpowers/specs/2026-08-26-signal-quality-and-summary-page-design.md.
"""
from dataclasses import dataclass, replace
import numpy as np
from canine_holter.types import Beat

WINDOW_SEC = 5.0
MAX_AMPLITUDE_RATIO = 4.0  # window peak-to-peak over this multiple of the median: off-body swings, saturation, gross motion
MIN_AMPLITUDE_RATIO = 0.1  # under this multiple: lead-off, flat line at a rail
MAX_FLAT_FRACTION = 0.9  # more than this share of zero sample-to-sample steps: flat line (a quiet DR200 baseline at 12.5 uV/count reaches ~0.5)
MAX_DIFFERENCE_POWER_RATIO = 3.0  # variance of sample-to-sample differences over variance of the window: 4 for a pure alternating signal, 2 for white noise, under 1.8 for ECG at 180 Hz. The front end's AC lead-off excitation at half the sample rate when an electrode is open.
EDGE_SEC = 60.0  # hookup and removal; the HE/LX vendor software calls the first and last minute artifact unconditionally
BRIDGE_SEC = 30.0  # excluded windows this close are one span: quiet stretches inside an off-body tail are not ECG either
PAD_SEC = 2.0  # beats right at a span's edge are half-buried in noise


@dataclass(frozen=True)
class SignalQuality:
    """duration_sec: length of the recording kept for analysis. excluded:
    (start, end) seconds of artifact, sorted, non-overlapping, clipped to
    the recording. trimmed_sec: off-body tail dropped from the end; the
    recorder ran duration_sec + trimmed_sec."""
    duration_sec: float
    excluded: tuple[tuple[float, float], ...]
    trimmed_sec: float = 0.0

    @property
    def analyzed_sec(self) -> float:
        return self.duration_sec - sum(end - start for start, end in self.excluded)

    def analyzed_within(self, start: float, end: float) -> float:
        """Seconds of [start, end) not excluded."""
        total = max(0.0, end - start)
        for s, e in self.excluded:
            total -= max(0.0, min(e, end) - max(s, start))
        return total

    def contains(self, t: float) -> bool:
        return any(s <= t <= e for s, e in self.excluded)


def assess_quality(samples: np.ndarray, sample_rate: float) -> SignalQuality:
    """Judge the recording in WINDOW_SEC windows; see the module docstring
    for the rules. A recording with no signal in any window (zero median
    peak-to-peak) is excluded whole rather than analyzed as flat. The
    remainder after the last full window needs no rule: the last-minute
    edge span always covers it."""
    duration = len(samples) / sample_rate
    if duration == 0:
        return SignalQuality(0.0, ())
    window = int(WINDOW_SEC * sample_rate)
    n = len(samples) // window
    if n == 0:  # shorter than one window: the edge rule covers all of it
        return SignalQuality(duration, ((0.0, duration),))
    windows = samples[: n * window].reshape(n, window)
    ptp = windows.max(axis=1) - windows.min(axis=1)
    flat_line = np.mean(np.diff(windows, axis=1) == 0, axis=1) > MAX_FLAT_FRACTION
    lead_off = _difference_power_ratio(windows) > MAX_DIFFERENCE_POWER_RATIO
    reference = ptp[~(lead_off | flat_line)]  # the absolute rules' windows would drag the median
    median = float(np.median(reference)) if reference.size else 0.0
    if median <= 0:
        return SignalQuality(duration, ((0.0, duration),))
    bad = (
        lead_off
        | flat_line
        | (ptp > MAX_AMPLITUDE_RATIO * median)
        | (ptp < MIN_AMPLITUDE_RATIO * median)
    )
    spans = [
        (max(0.0, i * WINDOW_SEC - PAD_SEC), min(duration, (i + 1) * WINDOW_SEC + PAD_SEC))
        for i in np.flatnonzero(bad)
    ]
    spans.append((0.0, min(EDGE_SEC, duration)))
    spans.append((max(0.0, duration - EDGE_SEC), duration))
    return SignalQuality(duration, _bridge(spans))


def _difference_power_ratio(windows: np.ndarray) -> np.ndarray:
    signal_var = windows.var(axis=1)
    diff_var = np.diff(windows, axis=1).var(axis=1)
    return np.divide(diff_var, signal_var, out=np.zeros_like(diff_var), where=signal_var > 0)


def _bridge(spans: list[tuple[float, float]]) -> tuple[tuple[float, float], ...]:
    """Merge overlapping spans and any separated by BRIDGE_SEC or less."""
    merged: list[list[float]] = []
    for start, end in sorted(spans):
        if merged and start - merged[-1][1] <= BRIDGE_SEC:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return tuple((s, e) for s, e in merged)


def exclude_beats(beats: list[Beat], quality: SignalQuality) -> list[Beat]:
    """Drop beats inside excluded spans. The first beat after each span
    gets rr_interval=None - the contract's "no previous beat" - so a span
    can never read as a pause, a run, or a sustained brady/tachy event."""
    kept: list[Beat] = []
    prev_time: float | None = None
    for beat in beats:
        if quality.contains(beat.time):
            continue
        # A span between this beat and the last kept one - whether or not
        # any beat was dropped inside it - means the RR crosses artifact.
        if prev_time is not None and any(s < beat.time and e > prev_time for s, e in quality.excluded):
            beat = replace(beat, rr_interval=None)
        kept.append(beat)
        prev_time = beat.time
    return kept
