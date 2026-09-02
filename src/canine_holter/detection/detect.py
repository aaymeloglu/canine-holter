import neurokit2 as nk
import numpy as np
from neurokit2.signal import signal_smooth
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from canine_holter.types import Beat

# QRS width: a Pan-Tompkins derivative-energy envelope, measured from the
# R-peak out to where it drops below a threshold on each side.
QRS_WIDTH_SEARCH_WINDOW_SEC = 0.15  # covers even a markedly wide ventricular QRS
QRS_ENVELOPE_INTEGRATION_SEC = 0.03
QRS_WIDTH_THRESHOLD_FRACTION = 0.1  # of the envelope at the R-peak, or ...
QRS_NOISE_FLOOR_FACTOR = 4.0  # ... this multiple of the local noise floor, whichever is higher: hash noise otherwise holds the envelope above 10% and the width lands on the noise, not the QRS edge
QRS_NOISE_FLOOR_CONTEXT_SEC = 1.0  # the floor is the envelope's median over +/- this; QRS complexes fill well under half of any second
# QRS bursts, as NeuroKit's default ("neurokit") detector finds them: the
# smoothed absolute gradient above BURST_THRESHOLD_WEIGHT times its
# BURST_AVERAGE_SEC running average, at least BURST_MIN_LENGTH_WEIGHT of
# the mean burst length, fiducials at least BURST_MIN_DELAY_SEC apart.
# NeuroKit keeps the most prominent local maximum inside a burst and drops
# a burst that has none; on a lead where the QRS is a small r and a deep S
# (DR400 Ch 2 and Ch 3 with the dog asleep) some beats have none, and the
# other leads cannot outvote a lead that saw nothing. See
# docs/superpowers/specs/2026-09-02-negative-qrs-fiducial-design.md.
BURST_AVERAGE_SEC = 0.75
BURST_THRESHOLD_WEIGHT = 1.5
BURST_MIN_LENGTH_WEIGHT = 0.4
BURST_MIN_DELAY_SEC = 0.3
BASELINE_SEC = 0.2  # a fiducial's deflection is measured from the median of this much signal before it
# Phantom beats: the detector invents beats inside long RR gaps at slow
# resting rates. A peak under MIN_R_AMPLITUDE_FRACTION of the median peak
# amplitude is one; a real PVC can be smaller than sinus beats, but not
# five times smaller on a working electrode. The window spans a whole
# QRS: on a negative QRS (small r, deep S) NeuroKit peaks on the r, and a
# window that stops short of the S reads a real beat as a phantom.
R_AMPLITUDE_WINDOW_SEC = QRS_WIDTH_SEARCH_WINDOW_SEC
MIN_R_AMPLITUDE_FRACTION = 0.2
# Search-back for beats missed at fast rates. NeuroKit's threshold is
# 1.5x a 0.75 s mean of the gradient; at ~150 bpm that mean rises until
# the threshold sits on the QRS. In fast rhythm a gap over GAP_FACTOR x
# the local RR is a missed beat; in slow rhythm it is sinus arrhythmia,
# so the pass is off there. LOCAL_RR_BEATS is the rhythm memory shared
# with the T-wave rule.
LOCAL_RR_BEATS = 8
FAST_RR_SEC = 0.8  # local median RR under this (>= 75 bpm) is fast rhythm
GAP_FACTOR = 1.5
FILL_FEATURE_FRACTION = 0.35  # candidate gradient feature vs the neighbouring beats' median
FILL_REFRACTORY_SEC = 0.25  # enforced after the fiducial is placed, against both neighbours
GRADIENT_SMOOTH_SEC = 0.1  # NeuroKit's own feature: |gradient| boxcar-smoothed
FIDUCIAL_HALF_SEC = 0.05  # the beat is the largest deflection from baseline this close to the steepest point
# T-wave rejection in slow rhythm. Lying down, a lead can show the QRS as
# a small spike and the T wave as a large trough 0.2-0.35 s later, past
# NeuroKit's 300 ms minimum spacing. The gradient feature cannot separate
# them (the broad T scores ~2x the spike), so the rule is timing: a
# candidate this soon after a beat, whose removal leaves the beat-to-beat
# interval equal to a neighbouring sinus interval, is interpolated - a T
# wave. A PVC that early resets the rhythm or is followed by a
# compensatory pause. Neighbouring intervals, not a running median,
# because resting sinus arrhythmia moves the RR by 30-50% within a few
# beats. Known cost: a genuinely interpolated R-on-T PVC at rest is
# dropped too.
SLOW_RR_SEC = 0.6  # only intervals over this (< 100 bpm) serve as the sinus reference: at 75-100 bpm the T wave still clears the 300 ms spacing, and tachycardia is excluded by the median gate
T_WAVE_MAX_COUPLING_SEC = 0.45
T_WAVE_RHYTHM_TOLERANCE = 0.25  # |A->C - reference| within this fraction of it means B was interpolated
# Lead agreement. A QRS is on every lead at once, so a beat is where at
# least MIN_AGREEING_LEADS leads detect a peak within
# AGREEMENT_TOLERANCE_SEC of each other; one lead's T wave, P wave, or
# noise spike has no partner. The same QRS's fiducial lands within ~20 ms
# on the other leads for most beats but up to 140 ms apart where a lead
# renders it as a fractured wiggle (the detector settles on its onset or
# the P wave). A T wave taken for a beat sits 250-350 ms after its QRS
# and the shortest RR seen is 255 ms at 235 bpm, so the tolerance is half
# the distance to the nearest distinct event.
AGREEMENT_TOLERANCE_SEC = 0.15
MIN_AGREEING_LEADS = 2


def detect_beats(leads: np.ndarray, sample_rate: float) -> list[Beat]:
    """Detect beats on one lead (a 1-D array) or several (n_leads, n_samples)
    and estimate each beat's QRS width, returning unlabeled Beats.

    Each lead is detected on its own: NeuroKit2's QRS bursts and R-peaks
    (_qrs_fiducials), then the amplitude, search-back, and T-wave passes
    below. With several leads a beat is where at least MIN_AGREEING_LEADS
    leads agree (see _agree), so a lead whose morphology has shifted with
    posture can neither invent a beat nor lose one the others see; a lead
    that shows a QRS as a bare trough can still vote for it. Its time is
    the median of the agreeing leads' peaks.

    QRS width comes from a Pan-Tompkins-style derivative-energy envelope
    measured on every lead at that lead's own peak, and the beat's width
    is the median across leads: with three leads, wide only when two are.
    NeuroKit's wave delineation was rejected for width because it fails
    on exactly the premature, wide beats a PVC classifier needs it for.

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


def _lead_peaks(cleaned: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """One lead's R-peaks, and a parallel boolean array marking the ones
    that came from the no-maximum fallback of _qrs_fiducials.

    The resolved peaks go through the three single-lead correction passes
    exactly as before. The fallback fiducials do not: they exist only to
    let this lead corroborate a beat another lead resolved (see _agree),
    and inside the passes they would fill gaps and set rhythm history that
    a T-wave trough has no business setting. A fallback within
    T_WAVE_MAX_COUPLING_SEC after a resolved peak is that beat's T wave,
    and one within BURST_MIN_DELAY_SEC before a resolved peak is part of
    that beat; both are dropped. The amplitude rule judges the rest against
    the resolved peaks' median."""
    resolved, fallback = _qrs_fiducials(cleaned, sample_rate)
    r_peaks = _reject_low_amplitude_peaks(cleaned, resolved, sample_rate)
    r_peaks = fill_fast_gaps(cleaned, r_peaks, sample_rate)
    r_peaks = drop_interpolated_t_waves(r_peaks, sample_rate)
    fallback = _reject_low_amplitude_peaks(cleaned, fallback, sample_rate, reference=resolved)
    after, before = int(T_WAVE_MAX_COUPLING_SEC * sample_rate), int(BURST_MIN_DELAY_SEC * sample_rate)
    fallback = np.array([
        f for f in fallback
        if not any(-before <= f - r <= after for r in r_peaks[max(0, np.searchsorted(r_peaks, f) - 1): np.searchsorted(r_peaks, f) + 1])
    ], dtype=int)
    peaks = np.sort(np.concatenate([r_peaks, fallback]))
    return peaks, np.isin(peaks, fallback)


def _qrs_fiducials(cleaned: np.ndarray, sample_rate: float) -> tuple[np.ndarray, np.ndarray]:
    """One fiducial per QRS burst (see the BURST_* constants), as two
    arrays: the bursts NeuroKit resolves and the bursts it drops.

    The burst logic is NeuroKit2's ``_ecg_findpeaks_neurokit`` (MIT
    licence), parameters and all: the first array is exactly what
    ``nk.ecg_peaks`` returns. The second holds a fiducial for each burst
    with no local maximum, at the _fiducial of its steepest sample. These
    fallback fiducials are for lead agreement to corroborate a beat another
    lead resolved (see _agree); on their own they are as often a negative T
    wave as a negative QRS, which is why they are kept apart."""
    smooth_kernel = int(np.rint(GRADIENT_SMOOTH_SEC * sample_rate))
    average_kernel = int(np.rint(BURST_AVERAGE_SEC * sample_rate))
    smooth_gradient = signal_smooth(np.abs(np.gradient(cleaned)), kernel="boxcar", size=smooth_kernel)
    average_gradient = signal_smooth(smooth_gradient, kernel="boxcar", size=average_kernel)
    burst = smooth_gradient > BURST_THRESHOLD_WEIGHT * average_gradient
    starts = np.where(~burst[:-1] & burst[1:])[0]
    ends = np.where(burst[:-1] & ~burst[1:])[0]
    empty = np.array([], dtype=int)
    if len(starts) == 0:
        return empty, empty
    ends = ends[ends > starts[0]]
    n = min(len(starts), len(ends))
    min_length = np.mean(ends[:n] - starts[:n]) * BURST_MIN_LENGTH_WEIGHT
    min_delay = int(np.rint(BURST_MIN_DELAY_SEC * sample_rate))
    resolved, fallback = [], []
    last_resolved = last_any = 0  # the delay for resolved peaks is against resolved peaks only, as in NeuroKit
    for start, end in zip(starts[:n], ends[:n]):
        if end - start < min_length:
            continue
        maxima, properties = find_peaks(cleaned[start:end], prominence=(None, None))
        if maxima.size > 0:
            fiducial = start + maxima[np.argmax(properties["prominences"])]
            if fiducial - last_resolved > min_delay:
                resolved.append(fiducial)
                last_resolved = last_any = fiducial
        else:
            fiducial = _fiducial(cleaned, start + int(np.argmax(smooth_gradient[start:end])), sample_rate)
            if fiducial - last_any > min_delay:
                fallback.append(fiducial)
                last_any = fiducial
    return np.asarray(resolved, dtype=int), np.asarray(fallback, dtype=int)


def _agree(peaks_by_lead: list[tuple[np.ndarray, np.ndarray]], tolerance: int) -> np.ndarray:
    """Beats that at least MIN_AGREEING_LEADS leads (or every lead, when
    there are fewer) detected in a chain of peaks each within `tolerance`
    samples of the next (see _clusters), at least one of them a resolved
    peak rather than a fallback, as an (n_beats, n_leads) array holding
    each lead's own peak, or the agreeing leads' median position for a
    lead that missed the beat. Each lead contributes (peaks, fallback
    flags) as _lead_peaks returns them."""
    n_leads = len(peaks_by_lead)
    needed = min(MIN_AGREEING_LEADS, n_leads)
    events = sorted(
        (int(p), k, bool(f)) for k, (peaks, flags) in enumerate(peaks_by_lead) for p, f in zip(peaks, flags)
    )
    rows = []
    for cluster in _clusters(events, tolerance):
        if len(cluster) >= needed and not all(fallback for _, fallback in cluster.values()):
            consensus = int(np.median([p for p, _ in cluster.values()]))
            rows.append([cluster.get(k, (consensus, False))[0] for k in range(n_leads)])
    return np.array(rows, dtype=int).reshape(-1, n_leads)


def _clusters(events: list[tuple[int, int, bool]], tolerance: int):
    """Group (position, lead, fallback) events, sorted by position, into
    chains whose consecutive events are within `tolerance` of each other
    and whose leads are distinct; each group maps lead -> (position,
    fallback). Chaining, rather than anchoring at the first event, keeps a
    QRS whose fiducials straddle the tolerance from splitting into two
    beats."""
    cluster: dict[int, tuple[int, bool]] = {}
    last = 0
    for position, lead, fallback in events:
        if cluster and (position - last > tolerance or lead in cluster):
            yield cluster
            cluster = {}
        cluster[lead] = (position, fallback)
        last = position
    if cluster:
        yield cluster


def _reject_low_amplitude_peaks(
    cleaned: np.ndarray, r_peaks: np.ndarray, sample_rate: float, reference: np.ndarray | None = None
) -> np.ndarray:
    """Drop detected peaks whose local peak-to-peak amplitude is far below
    the median over all peaks (or over `reference` peaks when given). A
    zero median means every peak sits on flat signal, so nothing is kept
    (fail closed)."""
    r_peaks = np.asarray(r_peaks, dtype=int)
    reference = r_peaks if reference is None else np.asarray(reference, dtype=int)
    if len(r_peaks) == 0 or len(reference) == 0:
        return r_peaks[:0]
    half = int(R_AMPLITUDE_WINDOW_SEC * sample_rate)

    def amplitude(peaks: np.ndarray) -> np.ndarray:
        return np.array([cleaned[max(0, r - half): r + half + 1].ptp() for r in peaks])

    median = np.median(amplitude(reference))
    if median <= 0:
        return r_peaks[:0]
    return r_peaks[amplitude(r_peaks) >= MIN_R_AMPLITUDE_FRACTION * median]


def _gradient_feature(cleaned: np.ndarray, sample_rate: float) -> np.ndarray:
    return uniform_filter1d(np.abs(np.gradient(cleaned)), max(1, int(GRADIENT_SMOOTH_SEC * sample_rate)))


def _fiducial(cleaned: np.ndarray, index: int, sample_rate: float) -> int:
    """The largest |deflection| from the preceding 200 ms baseline within
    FIDUCIAL_HALF_SEC of index - polarity-agnostic, so a negative QRS lands
    on its trough rather than its small r wave."""
    half = int(FIDUCIAL_HALF_SEC * sample_rate)
    lo, hi = max(0, index - half), min(len(cleaned), index + half + 1)
    baseline = np.median(cleaned[max(0, index - int(BASELINE_SEC * sample_rate)): index]) if index > 0 else 0.0
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
