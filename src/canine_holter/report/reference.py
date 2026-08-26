"""Published reference bands printed beside the report's numbers, and the
status (ok / caution / alert) each value gets from them.

PVC bands per 24 h: ESVC screening guidelines for dilated cardiomyopathy
in Doberman Pinschers (Wess et al., J Vet Cardiol 2017): under 50 normal,
50-300 equivocal (repeat within the year), over 300 abnormal; any couplet,
triplet, or run is abnormal. Run rate: ~180 bpm is the usual canine line
between ventricular tachycardia and the less concerning accelerated
idioventricular rhythm. Pauses: canine Holter studies of healthy dogs find
pauses over 2.5 s common with sinus arrhythmia; ~5 s is the usual line for
concern.
"""

PVC_24H_NORMAL_MAX = 50
PVC_24H_EQUIVOCAL_MAX = 300
VT_MIN_BPM = 180
PAUSE_COMMON_MAX_SEC = 2.5
PAUSE_CONCERN_SEC = 5.0
MIN_HOURS_FOR_24H_SCALING = 20  # PVC frequency varies across a day; don't scale a short recording
_SEC_PER_DAY = 24 * 3600.0

PVC_24H_BAND = f"<{PVC_24H_NORMAL_MAX} | {PVC_24H_NORMAL_MAX}-{PVC_24H_EQUIVOCAL_MAX} | >{PVC_24H_EQUIVOCAL_MAX}"
PAUSE_BAND = f"<{PAUSE_COMMON_MAX_SEC} | {PAUSE_COMMON_MAX_SEC}-{PAUSE_CONCERN_SEC:g} | >{PAUSE_CONCERN_SEC:g} s"
RUN_RATE_BAND = f"<{VT_MIN_BPM} bpm"
COUNT_BAND = "0"
ANALYZED_BAND = f">= {MIN_HOURS_FOR_24H_SCALING} h"

FOOTER_LINES = [  # kept under ~105 characters each: they print at 8 pt across the page
    "Colors compare each value with the band printed beside it: green inside the normal band, amber in the",
    "equivocal band, red beyond it. They are not a diagnosis.",
    "Bands: ESVC Doberman DCM screening guidelines (Wess et al., J Vet Cardiol 2017); pause and run-rate",
    "context from canine Holter studies.",
]


def format_duration(duration_sec: float) -> str:
    hours, rem = divmod(int(duration_sec), 3600)
    return f"{hours}h {rem // 60}m"


def pvc_per_24h(pvc_count: int, analyzed_sec: float) -> float | None:
    """PVC count scaled to 24 h of analyzed time, or None when fewer than
    MIN_HOURS_FOR_24H_SCALING hours were analyzed."""
    if analyzed_sec < MIN_HOURS_FOR_24H_SCALING * 3600:
        return None
    return pvc_count * _SEC_PER_DAY / analyzed_sec


def pvc_24h_status(scaled: float) -> str:
    if scaled < PVC_24H_NORMAL_MAX:
        return "ok"
    return "caution" if scaled <= PVC_24H_EQUIVOCAL_MAX else "alert"


def count_status(n: int) -> str:
    return "ok" if n == 0 else "alert"


def run_rate_status(bpm: float | None) -> str:
    if bpm is None:
        return "ok"
    return "alert" if bpm >= VT_MIN_BPM else "caution"


def pause_status(longest_sec: float | None) -> str:
    if longest_sec is None or longest_sec < PAUSE_COMMON_MAX_SEC:
        return "ok"
    return "caution" if longest_sec <= PAUSE_CONCERN_SEC else "alert"


def analyzed_status(analyzed_sec: float) -> str:
    return "ok" if analyzed_sec >= MIN_HOURS_FOR_24H_SCALING * 3600 else "caution"
