"""Published reference ranges printed beside the report's numbers, so a
reader can see how a count compares without the report calling anything
normal or abnormal.

PVC bands per 24 h: ESVC screening guidelines for dilated cardiomyopathy
in Doberman Pinschers (Wess et al., J Vet Cardiol 2017). Pause context:
canine Holter studies of healthy dogs, where pauses over 2.5 s with sinus
arrhythmia are common and ~5 s is the usual line for concern.
"""

PVC_24H_NORMAL_MAX = 50
PVC_24H_EQUIVOCAL_MAX = 300
MIN_HOURS_FOR_24H_SCALING = 20  # PVC frequency varies across a day; don't scale a short clip
_SEC_PER_DAY = 24 * 3600.0


def format_duration(duration_sec: float) -> str:
    hours, rem = divmod(int(duration_sec), 3600)
    return f"{hours}h {rem // 60}m"


def pvc_per_24h(pvc_count: int, duration_sec: float) -> float | None:
    """PVC count scaled to 24 h, or None when the recording is too short
    (under MIN_HOURS_FOR_24H_SCALING) for the scaled figure to mean much."""
    if duration_sec < MIN_HOURS_FOR_24H_SCALING * 3600:
        return None
    return pvc_count * _SEC_PER_DAY / duration_sec


def pvc_per_24h_line(pvc_count: int, duration_sec: float) -> str:
    scaled = pvc_per_24h(pvc_count, duration_sec)
    if scaled is None:
        return (
            f"- PVCs per 24 h: not computed (recording is {format_duration(duration_sec)};"
            f" needs >= {MIN_HOURS_FOR_24H_SCALING} h)"
        )
    return f"- PVCs per 24 h: {round(scaled)} (scaled from {format_duration(duration_sec)})"


def reference_lines(duration_sec: float) -> list[str]:
    """The Reference ranges block, one bullet per metric."""
    lines = [
        f"- PVCs per 24 h (Dobermans): under {PVC_24H_NORMAL_MAX} is considered normal, though any"
        " PVCs merit attention;",
        f"  {PVC_24H_NORMAL_MAX}-{PVC_24H_EQUIVOCAL_MAX} is equivocal and a repeat Holter within the"
        f" year is advised; over {PVC_24H_EQUIVOCAL_MAX} is considered abnormal.",
        "- Couplets, triplets, or VT runs: any is worth a cardiologist's review, whatever the PVC count.",
        "- Pauses: pauses over 2.5 s are common in healthy dogs with sinus arrhythmia, especially at rest;",
        "  pauses over ~5 s, or any pause alongside fainting or collapse, warrant review.",
        "- Source: ESVC Doberman DCM screening guidelines (Wess et al., J Vet Cardiol 2017).",
    ]
    if pvc_per_24h(0, duration_sec) is None:
        lines.append(
            f"- This recording is {format_duration(duration_sec)}, shorter than 24 h, so the PVC"
            " band does not apply to it."
        )
    return lines
