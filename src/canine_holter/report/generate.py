import os
import matplotlib
matplotlib.use("Agg")  # no display needed - this runs headless in CLI/CI
import matplotlib.pyplot as plt
import numpy as np
from canine_holter.types import Beat
from canine_holter.arrhythmia.burden import ArrhythmiaSummary, pvc_runs

STRIP_WINDOW_SEC = 6.0  # seconds of context shown around each flagged run


def _flagged_runs(beats: list[Beat]) -> list[list[Beat]]:
    """PVC runs of 2+ beats (couplets, triplets, VT runs) - the events worth
    a rhythm-strip plot. Isolated single PVCs are counted in the summary
    stats but not individually plotted, to avoid an unbounded number of
    plots on a high-burden recording."""
    return [run for run in pvc_runs(beats) if len(run) >= 2]


def _plot_strip(samples: np.ndarray, sample_rate: float, center_time: float, out_path: str) -> None:
    half_window = STRIP_WINDOW_SEC / 2
    start_sample = max(0, int((center_time - half_window) * sample_rate))
    end_sample = min(len(samples), int((center_time + half_window) * sample_rate))
    segment = samples[start_sample:end_sample]
    t = np.arange(len(segment)) / sample_rate

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(t, segment, linewidth=0.8)
    ax.set_title(f"Rhythm strip around t={center_time:.1f}s")
    ax.set_xlabel("seconds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_report(
    beats: list[Beat],
    summary: ArrhythmiaSummary,
    out_dir: str,
    samples: np.ndarray | None,
    sample_rate: float | None,
) -> str:
    """Write a markdown summary report plus (if waveform data is provided)
    rhythm-strip PNGs for each flagged multi-beat PVC run. Returns the path
    to the markdown report."""
    os.makedirs(out_dir, exist_ok=True)

    lines = [
        "# Holter Analysis Report",
        "",
        "**This is a screening aid, not a diagnosis. Review with a veterinary cardiologist.**",
        "",
        "## Summary",
        f"- Total beats: {summary.total_beats}",
        f"- PVC count: {summary.pvc_count}",
        f"- PVC burden: {summary.pvc_burden_pct:.2f}%",
        f"- Couplets: {summary.couplets}",
        f"- Triplets: {summary.triplets}",
        f"- VT runs (4+ consecutive PVCs): {summary.vtach_runs}",
        f"- Pauses (>= threshold): {len(summary.pauses)}",
        f"- Sustained bradycardia events: {len(summary.bradycardia_events)}",
        f"- Sustained tachycardia events: {len(summary.tachycardia_events)}",
        "",
    ]

    flagged = _flagged_runs(beats)
    if flagged:
        lines.append("## Flagged events (couplets, triplets, VT runs)")
        for i, run in enumerate(flagged):
            center_time = (run[0].time + run[-1].time) / 2
            lines.append(f"- Event {i + 1}: {len(run)} consecutive PVCs at ~t={center_time:.1f}s")
            if samples is not None and sample_rate is not None:
                plot_path = os.path.join(out_dir, f"event_{i + 1}_strip.png")
                _plot_strip(samples, sample_rate, center_time, plot_path)
                lines.append(f"  ![event {i + 1}]({os.path.basename(plot_path)})")
        lines.append("")

    report_path = os.path.join(out_dir, "report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))

    return report_path
