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
# The crossing threshold must also clear the local noise floor: the median
# of the envelope over the surrounding +/-1 s (QRS complexes occupy well
# under half of any second). Hash noise otherwise holds the envelope above
# 10% of the peak and the width lands on the noise, not the QRS edge - 62
# of 93 "PVCs" on Teeny's 2026-08-25 report were normal beats measured
# 3-5 samples wide of baseline that way.
QRS_NOISE_FLOOR_FACTOR = 4.0
QRS_NOISE_FLOOR_CONTEXT_SEC = 1.0
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
# whose removal leaves the beat-to-beat interval equal to a neighbouring
# sinus interval, is interpolated - a T wave. A PVC that early resets the
# rhythm or is followed by a compensatory pause. The comparison is with
# the neighbouring intervals, not a running median, because resting sinus
# arrhythmia moves the RR by 30-50% within a few beats. Known cost: a
# genuinely interpolated R-on-T PVC at rest is dropped too.
SLOW_RR_SEC = 0.6  # only intervals over this (< 100 bpm) serve as the sinus reference; at 75-100 bpm the T wave still clears NeuroKit's 300 ms spacing, and tachycardia is excluded by the median gate
T_WAVE_MAX_COUPLING_SEC = 0.45
T_WAVE_RHYTHM_TOLERANCE = 0.25  # |A->C - reference| within this fraction of it means B was interpolated
# Lead agreement. A QRS is on every lead at once, so a beat is where at
# least MIN_AGREEING_LEADS leads detect a peak within AGREEMENT_TOLERANCE_SEC
# of each other; one lead's T wave, P wave, or noise spike has no partner.
# On Teeny's 2026-08-27 recording the same QRS's fiducial lands within
# 22 ms on the other leads for 90% of beats, but up to 140 ms apart where
# a lead renders the QRS as a fractured wiggle (channel 1 asleep; the
# detector settles on its onset or the P wave). A T wave taken for a beat
# sits 250-350 ms after its QRS and the shortest RR seen is 255 ms at
# 235 bpm, so the tolerance is half the distance to the nearest distinct
# event.
AGREEMENT_TOLERANCE_SEC = 0.15
MIN_AGREEING_LEADS = 2


def detect_beats(leads: np.ndarray, sample_rate: float) -> list[Beat]:
    """Detect beats on one lead (a 1-D array) or several (n_leads, n_samples)
    and estimate each beat's QRS width, returning unlabeled Beats.

    Each lead is detected on its own: NeuroKit2 R-peaks, then the
    amplitude, search-back, and T-wave passes below. With several leads a
    beat is where at least MIN_AGREEING_LEADS leads agree (see _agree), so
    a lead whose morphology has shifted with posture can neither invent a
    beat nor lose one the others see. Its time is the median of the
    agreeing leads' peaks.

    QRS width comes from a Pan-Tompkins-style derivative-energy envelope
    measured on every lead at that lead's own peak, and the beat's width
    is the median across leads: with three leads, wide only when two are.
    Delineation-based onset/offset detection (`ecg_delineate`) was
    rejected because it fails on exactly the premature/wide beats a PVC
    classifier needs width for; validated on MIT-BIH 119 (65/65 beats
    measured, no overlap between normal 0.069-0.078 s and PVC 0.158-0.183 s).

    Specs: docs/superpowers/specs/2026-08-26-detector-tachycardia-and-t-wave-design.md,
    docs/superpowers/specs/2026-09-01-lead-agreement-qrs-width-design.md.
    """
    leads = np.atleast_2d(np.asarray(leads, dtype=float))
    cleaned = [nk.ecg_clean(lead, sampling_rate=sample_rate) for lead in leads]
    positions = _agree(
        [_lead_peaks(lead, sample_rate) for lead in cleaned],
        int(AGREEMENT_TOLERANCE_SEC * sample_rate),
    )
    if len(positions) < 2:
        return []

    envelopes = [_qrs_energy_envelope(lead, sample_rate) for lead in cleaned]
    search_half = int(QRS_WIDTH_SEARCH_WINDOW_SEC * sample_rate)
    times = np.median(positions, axis=1) / sample_rate

    beats = []
    for i, row in enumerate(positions):
        widths = [
            w for envelope, r in zip(envelopes, row)
            if (w := _qrs_width(envelope, int(r), search_half, sample_rate)) is not None
        ]
        beats.append(Beat(
            time=float(times[i]),
            rr_interval=float(times[i] - times[i - 1]) if i > 0 else None,
            qrs_duration=float(np.median(widths)) if widths else None,
            label=None,
        ))
    return beats


def _lead_peaks(cleaned: np.ndarray, sample_rate: float) -> np.ndarray:
    """One lead's R-peaks after the three single-lead correction passes."""
    _, r_info = nk.ecg_peaks(cleaned, sampling_rate=sample_rate)
    r_peaks = _reject_low_amplitude_peaks(cleaned, r_info["ECG_R_Peaks"], sample_rate)
    r_peaks = fill_fast_gaps(cleaned, r_peaks, sample_rate)
    return drop_interpolated_t_waves(r_peaks, sample_rate)


def _agree(peaks_by_lead: list[np.ndarray], tolerance: int) -> np.ndarray:
    """Beats that at least MIN_AGREEING_LEADS leads (or every lead, when
    there are fewer) detected in a chain of peaks each within `tolerance`
    samples of the next (see _clusters), as
    an (n_beats, n_leads) array holding each lead's own peak, or the
    agreeing leads' median position for a lead that missed the beat."""
    n_leads = len(peaks_by_lead)
    needed = min(MIN_AGREEING_LEADS, n_leads)
    rows = []
    for cluster in _clusters(sorted((int(p), k) for k, ps in enumerate(peaks_by_lead) for p in ps), tolerance):
        if len(cluster) >= needed:
            consensus = int(np.median(list(cluster.values())))
            rows.append([cluster.get(k, consensus) for k in range(n_leads)])
    return np.array(rows, dtype=int).reshape(-1, n_leads)


def _clusters(events: list[tuple[int, int]], tolerance: int):
    """Group (position, lead) events, sorted by position, into chains whose
    consecutive events are within `tolerance` of each other and whose
    leads are distinct; each group maps lead -> position. Anchoring a
    group at its first event instead split one QRS in two when the leads'
    fiducials straddled the tolerance, and the second half read as a
    premature wide beat 160 ms after the first."""
    cluster: dict[int, int] = {}
    last = 0
    for position, lead in events:
        if cluster and (position - last > tolerance or lead in cluster):
            yield cluster
            cluster = {}
        cluster[lead] = position
        last = position
    if cluster:
        yield cluster


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
    rhythm when the interval A -> C (C the next peak) matches a neighbouring
    slow sinus interval - the accepted interval before A, or C to the next
    peak more than T_WAVE_MAX_COUPLING_SEC after C (so C's own T wave is
    skipped). B is then a T wave interpolated into an undisturbed rhythm.
    Slow rhythm is the median of at least three of the last LOCAL_RR_BEATS
    accepted intervals over SLOW_RR_SEC, and it is mandatory: without it, in
    fast rhythm the look-ahead that skips C's T wave skips the real next
    beat and a double interval matches A -> C. Sequential and causal in the
    accepted intervals."""
    peaks = np.asarray(peaks, dtype=int)
    times = peaks / sample_rate
    keep = np.ones(len(times), dtype=bool)
    history: list[float] = []
    i = 1
    while i < len(times) - 1:
        a, b, c = times[i - 1], times[i], times[i + 1]
        slow = len(history) >= 3 and float(np.median(history[-LOCAL_RR_BEATS:])) > SLOW_RR_SEC
        following = next((t for t in times[i + 2:] if t - c > T_WAVE_MAX_COUPLING_SEC), None)
        references = [rr for rr in (history[-1] if history else None, following - c if following is not None else None)
                      if rr is not None and rr > SLOW_RR_SEC]
        if slow and (b - a) < T_WAVE_MAX_COUPLING_SEC and any(
            abs((c - a) - rr) < T_WAVE_RHYTHM_TOLERANCE * rr for rr in references
        ):
            keep[i] = False
            history.append(c - a)
            i += 2
            continue
        history.append(b - a)
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

    The threshold is QRS_WIDTH_THRESHOLD_FRACTION of the envelope at the
    peak or QRS_NOISE_FLOOR_FACTOR times the local noise floor, whichever
    is higher. Returns None if the R-peak has no measurable energy, if the
    noise floor reaches the peak itself (a beat buried in noise), or if the
    envelope never drops back below threshold within the search window on
    either side (e.g. a beat too close to the start/end of the recording).
    """
    lo = max(0, r_peak - search_half)
    hi = min(len(envelope), r_peak + search_half)
    local_peak = envelope[r_peak]
    if local_peak <= 0:
        return None
    context_half = int(QRS_NOISE_FLOOR_CONTEXT_SEC * sample_rate)
    noise_floor = float(np.median(envelope[max(0, r_peak - context_half): r_peak + context_half]))
    threshold = max(QRS_WIDTH_THRESHOLD_FRACTION * local_peak, QRS_NOISE_FLOOR_FACTOR * noise_floor)
    if threshold >= local_peak:
        return None

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
