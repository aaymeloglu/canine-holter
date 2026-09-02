import statistics
from collections import deque
from dataclasses import replace
from canine_holter.types import Beat

PREMATURITY_RATIO = 0.85  # RR < 85% of local baseline -> premature
QRS_WIDTH_RATIO = 1.25  # QRS > 125% of local baseline -> wide ...
QRS_WIDTH_MARGIN_SEC = 0.030  # ... and at least this much wider: 1.25x is three samples at 180 Hz; one small square on paper is 40 ms
ESCAPE_RR_RATIO = 1.5  # RR >= 150% of local baseline -> late: a wide beat this late is the ventricle escaping after a gap, not a premature beat. Resting sinus arrhythmia runs to 1.1-1.4x; see the 2026-09-02 escape-beat spec
BASELINE_WINDOW = 8  # number of recent "N" beats used to compute baseline


def classify_beats(beats: list[Beat]) -> list[Beat]:
    """Label each beat "N" (normal), "V" (PVC), "E" (ventricular escape
    beat), or "U" (undetermined).

    A beat is "V" only when it is BOTH premature (RR well below the local
    baseline) AND wide (QRS above the local baseline by QRS_WIDTH_RATIO and
    by at least QRS_WIDTH_MARGIN_SEC - a ratio alone is sample jitter at
    180 Hz) - this matches the standard clinical heuristic for identifying
    ventricular ectopy. A beat that is wide and instead LATE (RR at least
    ESCAPE_RR_RATIO times the baseline) is "E": the ventricle stepping in
    after a gap the sinus node left. A wide beat at ordinary timing stays
    "N"; width alone is too noisy a measurement to flag on.
    Baseline is computed causally from the most recent beats labeled "N",
    so thresholds adapt to each recording's (and each dog's) own rhythm
    rather than relying on a fixed literature value; "V" and "E" beats
    never feed it.

    Bootstrapping: before any baseline exists, there is nothing to compare
    a beat's RR/QRS against, so a beat with complete measurements is
    provisionally labeled "N" (this is what seeds the baseline). A beat is
    only "U" (undetermined) when its own RR interval or QRS duration is
    missing - never as a substitute for "we haven't decided yet".

    IMPORTANT - causality invariant: the loop below must remain strictly
    sequential. Beat N's classification may only use beats 0..N-1 (via
    baseline_rr/baseline_qrs, which are only updated *after* a beat is
    labeled). Do not replace this loop with a vectorized/batch rolling-median
    computation (e.g. a centered pandas rolling window) - that would let
    future beats leak into a beat's baseline, which is invisible in offline
    testing but wrong for a real-time classifier.
    """
    baseline_rr: deque[float] = deque(maxlen=BASELINE_WINDOW)
    baseline_qrs: deque[float] = deque(maxlen=BASELINE_WINDOW)
    labeled: list[Beat] = []

    for beat in beats:
        label = "U"

        have_baseline = len(baseline_rr) > 0 and len(baseline_qrs) > 0
        have_measurements = beat.rr_interval is not None and beat.qrs_duration is not None

        if have_measurements:
            if have_baseline:
                rr_base = statistics.median(baseline_rr)
                qrs_base = statistics.median(baseline_qrs)
                is_premature = beat.rr_interval < PREMATURITY_RATIO * rr_base
                is_late = beat.rr_interval >= ESCAPE_RR_RATIO * rr_base
                is_wide = (
                    beat.qrs_duration > QRS_WIDTH_RATIO * qrs_base
                    and beat.qrs_duration - qrs_base >= QRS_WIDTH_MARGIN_SEC
                )
                if is_wide and is_premature:
                    label = "V"
                elif is_wide and is_late:
                    label = "E"
                else:
                    label = "N"
            else:
                # No baseline yet: nothing to measure prematurity/width
                # against, so this beat can't be flagged as a PVC. Treat it
                # as provisionally normal so the baseline can seed itself.
                label = "N"

        labeled_beat = replace(beat, label=label)
        labeled.append(labeled_beat)

        if label == "N":
            if beat.rr_interval is not None:
                baseline_rr.append(beat.rr_interval)
            if beat.qrs_duration is not None:
                baseline_qrs.append(beat.qrs_duration)

    return labeled
