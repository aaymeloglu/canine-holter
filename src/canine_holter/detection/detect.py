import neurokit2 as nk
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from canine_holter.types import Beat

# How far each side of an R-peak to search for the QRS envelope crossing
# back below threshold. 150ms comfortably covers even markedly wide
# (aberrant/ventricular) QRS complexes in both human and canine ECG.
QRS_WIDTH_SEARCH_WINDOW_SEC = 0.15
# Moving-window length used to integrate the squared derivative into an
# energy envelope (classic Pan-Tompkins preprocessing).
QRS_ENVELOPE_INTEGRATION_SEC = 0.03
# Envelope must drop below this fraction of its value at the R-peak to
# mark the QRS onset/offset boundary.
QRS_WIDTH_THRESHOLD_FRACTION = 0.1
# Peak-to-peak range within this window of a detected R-peak is its
# amplitude; a peak under MIN_R_AMPLITUDE_FRACTION of the recording's
# median R amplitude is a phantom (the detector inventing a beat inside a
# long RR gap at slow resting rates), not a beat. Real PVCs can be smaller
# than sinus beats, but not five times smaller on a working electrode.
# The window must span a whole QRS, not just the detector's peak: on a
# negative QRS (small r, deep S) NeuroKit puts the peak on the r, and a
# window that stops short of the S reads a real beat as a phantom.
R_AMPLITUDE_WINDOW_SEC = QRS_WIDTH_SEARCH_WINDOW_SEC
MIN_R_AMPLITUDE_FRACTION = 0.2
# Search-back for beats the detector missed at fast rates. NeuroKit's
# threshold is 1.5x a 0.75 s mean of the gradient; at ~150 bpm that mean
# rises until the threshold sits on the QRS and beats are lost in
# fragments. In fast rhythm a gap over GAP_FACTOR x the local RR is a missed
# beat; in slow rhythm it is sinus arrhythmia, so the pass is off there
# (and T waves, which sit inside the refractory at fast rates, cannot be
# filled in). LOCAL_RR_BEATS is the rhythm memory shared with the T-wave rule.
LOCAL_RR_BEATS = 8
FAST_RR_SEC = 0.8  # local median RR under this (>= 75 bpm) is "fast rhythm"
GAP_FACTOR = 1.5
FILL_FEATURE_FRACTION = 0.35  # candidate gradient feature vs the neighbouring beats' median
FILL_REFRACTORY_SEC = 0.25  # enforced after the fiducial is placed, against both neighbours
GRADIENT_SMOOTH_SEC = 0.1  # NeuroKit's own feature: |gradient| boxcar-smoothed
FIDUCIAL_HALF_SEC = 0.05  # the beat is the largest deflection from baseline this close to the steepest point
# T-wave rejection in slow rhythm. Lying down, Teeny's analysis lead shows
# the QRS as a small spike and the T wave as a large trough 0.2-0.35 s
# later; past NeuroKit's 300 ms minimum spacing the T is detected as a
# beat. The gradient feature cannot separate them (the broad T scores ~2x
# the spike), so the rule is timing: a candidate this soon after a beat,
# whose removal leaves the beat-to-beat interval equal to the local RR, is
# interpolated - a T wave. A PVC that early resets the rhythm or is
# followed by a compensatory pause. Known cost: a genuinely interpolated
# R-on-T PVC at rest is dropped too.
SLOW_RR_SEC = 0.8  # local median RR over this (< 75 bpm) is "slow rhythm"
T_WAVE_MAX_COUPLING_SEC = 0.45
T_WAVE_RHYTHM_TOLERANCE = 0.25  # |A->C - local RR| within this fraction means B was interpolated


def detect_beats(samples: np.ndarray, sample_rate: float) -> list[Beat]:
    """Detect R-peaks and estimate QRS width, returning unlabeled Beats.

    QRS width comes from a Pan-Tompkins-style derivative-energy envelope
    measured directly around each R-peak, not NeuroKit2's wave delineation
    (`ecg_delineate`). Delineation-based onset/offset detection is tuned for
    normal QRS morphology and reliably fails (NaN, or a method-dependent
    fixed-window cap) on exactly the premature/wide beats a PVC classifier
    needs width for. Validated on MIT-BIH record 119: 65/65 beats get a
    valid width, with zero overlap between normal (0.069-0.078s) and PVC
    (0.158-0.183s) ranges, vs. the original delineation-based approach
    which returned NaN for 19/19 ground-truth PVC beats.
    """
    cleaned = nk.ecg_clean(samples, sampling_rate=sample_rate)
    _, r_info = nk.ecg_peaks(cleaned, sampling_rate=sample_rate)
    r_peaks = _reject_low_amplitude_peaks(cleaned, r_info["ECG_R_Peaks"], sample_rate)
    if len(r_peaks) < 2:
        return []

    envelope = _qrs_energy_envelope(cleaned, sample_rate)
    search_half = int(QRS_WIDTH_SEARCH_WINDOW_SEC * sample_rate)

    beats = []
    for i, r in enumerate(r_peaks):
        time = r / sample_rate
        rr = (r - r_peaks[i - 1]) / sample_rate if i > 0 else None
        qrs_duration = _qrs_width(envelope, r, search_half, sample_rate)
        beats.append(
            Beat(time=time, rr_interval=rr, qrs_duration=qrs_duration, label=None)
        )
    return beats


def _reject_low_amplitude_peaks(
    cleaned: np.ndarray, r_peaks: np.ndarray, sample_rate: float
) -> np.ndarray:
    """Drop detected peaks whose local peak-to-peak amplitude is far below
    the median over all peaks. A zero median means every peak sits on flat
    signal, so nothing is kept (fail closed)."""
    r_peaks = np.asarray(r_peaks, dtype=int)
    if len(r_peaks) == 0:
        return r_peaks
    half = int(R_AMPLITUDE_WINDOW_SEC * sample_rate)
    amplitudes = np.array([
        cleaned[max(0, r - half): r + half + 1].ptp() for r in r_peaks
    ])
    reference = np.median(amplitudes)
    if reference <= 0:
        return r_peaks[:0]
    return r_peaks[amplitudes >= MIN_R_AMPLITUDE_FRACTION * reference]


def _gradient_feature(cleaned: np.ndarray, sample_rate: float) -> np.ndarray:
    return uniform_filter1d(np.abs(np.gradient(cleaned)), max(1, int(GRADIENT_SMOOTH_SEC * sample_rate)))


def _fiducial(cleaned: np.ndarray, index: int, sample_rate: float) -> int:
    """The largest |deflection| from the preceding 200 ms baseline within
    FIDUCIAL_HALF_SEC of index - polarity-agnostic, so a negative QRS lands
    on its trough rather than its small r wave."""
    half = int(FIDUCIAL_HALF_SEC * sample_rate)
    lo, hi = max(0, index - half), min(len(cleaned), index + half + 1)
    baseline = np.median(cleaned[max(0, index - int(0.2 * sample_rate)): index]) if index > 0 else 0.0
    return lo + int(np.argmax(np.abs(cleaned[lo:hi] - baseline)))


def fill_fast_gaps(cleaned: np.ndarray, peaks: np.ndarray, sample_rate: float) -> np.ndarray:
    """Add beats inside gaps that are implausible for a fast local rhythm.

    Only acts when the median of the previous LOCAL_RR_BEATS RRs is under
    FAST_RR_SEC and the gap exceeds GAP_FACTOR times it. Candidates are
    peaks of the gradient feature at least FILL_FEATURE_FRACTION of the
    surrounding beats' feature, placed at their fiducial, at least
    FILL_REFRACTORY_SEC from the previous accepted peak and from the gap's end.
    """
    peaks = np.asarray(peaks, dtype=int)
    if len(peaks) < 2:
        return peaks
    feature = _gradient_feature(cleaned, sample_rate)
    half = int(GRADIENT_SMOOTH_SEC * sample_rate)
    peak_feature = np.array([feature[max(0, p - half): p + half + 1].max() for p in peaks])
    refractory = int(FILL_REFRACTORY_SEC * sample_rate)
    added = []
    for i in range(1, len(peaks)):
        a, b = peaks[i - 1], peaks[i]
        previous_rr = np.diff(peaks[max(0, i - 1 - LOCAL_RR_BEATS): i]) / sample_rate
        if len(previous_rr) < 3:
            continue
        local_rr = float(np.median(previous_rr))
        if local_rr >= FAST_RR_SEC or (b - a) / sample_rate <= GAP_FACTOR * local_rr:
            continue
        reference = float(np.median(peak_feature[max(0, i - 1 - LOCAL_RR_BEATS): i + 1]))
        lo, hi = a + refractory, b - refractory
        if hi <= lo:
            continue
        candidates, _ = find_peaks(feature[lo:hi], height=FILL_FEATURE_FRACTION * reference, distance=refractory)
        last = a
        for candidate in candidates:
            fiducial = _fiducial(cleaned, lo + candidate, sample_rate)
            if fiducial - last >= refractory and b - fiducial >= refractory:
                added.append(fiducial)
                last = fiducial
    return np.array(sorted(set(peaks.tolist()) | set(added)), dtype=int)


def drop_interpolated_t_waves(peaks: np.ndarray, sample_rate: float) -> np.ndarray:
    """Drop a peak B that follows A within T_WAVE_MAX_COUPLING_SEC in slow
    rhythm when the next peak C sits one local RR after A - B is a T wave
    interpolated into an undisturbed rhythm. Sequential and causal: the
    local RR is the median of the last LOCAL_RR_BEATS accepted intervals."""
    peaks = np.asarray(peaks, dtype=int)
    times = peaks / sample_rate
    keep = np.ones(len(times), dtype=bool)
    rr_history: list[float] = []
    i = 1
    while i < len(times) - 1:
        a, b, c = times[i - 1], times[i], times[i + 1]
        local_rr = float(np.median(rr_history[-LOCAL_RR_BEATS:])) if len(rr_history) >= 3 else None
        if (
            local_rr is not None
            and local_rr > SLOW_RR_SEC
            and (b - a) < T_WAVE_MAX_COUPLING_SEC
            and abs((c - a) - local_rr) < T_WAVE_RHYTHM_TOLERANCE * local_rr
        ):
            keep[i] = False
            rr_history.append(c - a)
            i += 2
            continue
        rr_history.append(b - a)
        i += 1
    return peaks[keep]


def _qrs_energy_envelope(cleaned: np.ndarray, sample_rate: float) -> np.ndarray:
    """Derivative-squared, moving-window-integrated energy envelope."""
    derivative = np.diff(cleaned, prepend=cleaned[0])
    squared = derivative**2
    window_samples = max(1, int(QRS_ENVELOPE_INTEGRATION_SEC * sample_rate))
    kernel = np.ones(window_samples) / window_samples
    return np.convolve(squared, kernel, mode="same")


def _qrs_width(
    envelope: np.ndarray, r_peak: int, search_half: int, sample_rate: float
) -> float | None:
    """Width between the envelope's threshold crossings on either side of r_peak.

    Returns None if the R-peak has no measurable energy, or if the envelope
    never drops back below threshold within the search window on either side
    (e.g. a beat too close to the start/end of the recording).
    """
    lo = max(0, r_peak - search_half)
    hi = min(len(envelope), r_peak + search_half)
    local_peak = envelope[r_peak]
    if local_peak <= 0:
        return None
    threshold = QRS_WIDTH_THRESHOLD_FRACTION * local_peak

    onset = None
    for i in range(r_peak, lo - 1, -1):
        if envelope[i] < threshold:
            onset = i
            break
    offset = None
    for i in range(r_peak, hi):
        if envelope[i] < threshold:
            offset = i
            break
    if onset is None or offset is None:
        return None
    return (offset - onset) / sample_rate
