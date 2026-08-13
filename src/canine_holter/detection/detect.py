import math
import neurokit2 as nk
import numpy as np
from canine_holter.types import Beat


def detect_beats(samples: np.ndarray, sample_rate: float) -> list[Beat]:
    """Detect R-peaks and delineate QRS onset/offset, returning unlabeled Beats."""
    cleaned = nk.ecg_clean(samples, sampling_rate=sample_rate)
    _, r_info = nk.ecg_peaks(cleaned, sampling_rate=sample_rate)
    r_peaks = r_info["ECG_R_Peaks"]
    if len(r_peaks) < 2:
        return []

    _, waves = nk.ecg_delineate(
        cleaned, r_peaks, sampling_rate=sample_rate, method="dwt"
    )
    onsets = waves.get("ECG_R_Onsets", [None] * len(r_peaks))
    offsets = waves.get("ECG_R_Offsets", [None] * len(r_peaks))

    beats = []
    for i, r in enumerate(r_peaks):
        time = r / sample_rate
        rr = (r - r_peaks[i - 1]) / sample_rate if i > 0 else None

        qrs_duration = None
        if i < len(onsets) and i < len(offsets):
            onset, offset = onsets[i], offsets[i]
            if onset is not None and offset is not None:
                if not (math.isnan(onset) or math.isnan(offset)):
                    qrs_duration = (offset - onset) / sample_rate

        beats.append(
            Beat(time=time, rr_interval=rr, qrs_duration=qrs_duration, label=None)
        )
    return beats
