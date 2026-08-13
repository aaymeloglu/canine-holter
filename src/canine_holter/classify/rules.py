import statistics
from collections import deque
from dataclasses import replace
from canine_holter.types import Beat

PREMATURITY_RATIO = 0.85  # RR < 85% of local baseline -> premature
QRS_WIDTH_RATIO = 1.25  # QRS > 125% of local baseline -> wide
BASELINE_WINDOW = 8  # number of recent "N" beats used to compute baseline


def classify_beats(beats: list[Beat]) -> list[Beat]:
    """Label each beat "N" (normal), "V" (PVC), or "U" (undetermined).

    A beat is "V" only when it is BOTH premature (RR well below the local
    baseline) AND wide (QRS well above the local baseline) - this matches
    the standard clinical heuristic for identifying ventricular ectopy.
    Baseline is computed causally from the most recent beats labeled "N",
    so thresholds adapt to each recording's (and each dog's) own rhythm
    rather than relying on a fixed literature value.

    Bootstrapping: before any baseline exists, there is nothing to compare
    a beat's RR/QRS against, so a beat with complete measurements is
    provisionally labeled "N" (this is what seeds the baseline). A beat is
    only "U" (undetermined) when its own RR interval or QRS duration is
    missing - never as a substitute for "we haven't decided yet".
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
                is_wide = beat.qrs_duration > QRS_WIDTH_RATIO * qrs_base
                label = "V" if (is_premature and is_wide) else "N"
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
