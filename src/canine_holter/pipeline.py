# src/canine_holter/pipeline.py
from canine_holter.ingest.wfdb_loader import load_local_record
from canine_holter.detection.detect import detect_beats
from canine_holter.classify.rules import classify_beats
from canine_holter.arrhythmia.burden import summarize
from canine_holter.report.generate import write_report


def run_analysis(input_path: str, out_dir: str, dog_weight_class: str = "medium") -> str:
    """Run the full ingest -> detect -> classify -> summarize -> report
    pipeline against a local WFDB record. Returns the path to the written
    markdown report.

    NOTE: input_path currently must be a local WFDB record (see
    ingest/wfdb_loader.py). DR200-native files aren't supported yet - see
    docs/superpowers/specs/2026-08-13-pvc-detection-design.md, "Open items".
    """
    rec = load_local_record(input_path, source=input_path)
    beats = detect_beats(rec.samples, rec.sample_rate)
    labeled = classify_beats(beats)
    summary = summarize(labeled, dog_weight_class=dog_weight_class)
    return write_report(
        labeled, summary, out_dir, samples=rec.samples, sample_rate=rec.sample_rate
    )
