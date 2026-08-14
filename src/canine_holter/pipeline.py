# src/canine_holter/pipeline.py
from canine_holter.ingest.loader import load_recording
from canine_holter.detection.detect import detect_beats
from canine_holter.classify.rules import classify_beats
from canine_holter.arrhythmia.burden import summarize
from canine_holter.report.generate import write_report


def run_analysis(input_path: str, out_dir: str, dog_weight_class: str = "medium") -> str:
    """Run the full ingest -> detect -> classify -> summarize -> report
    pipeline against a supported recording. Returns the path to the written
    markdown report. WFDB records, native DR200 flash.dat recordings, and
    vendor-extracted DR200 channel files are supported.
    """
    rec = load_recording(input_path)
    beats = detect_beats(rec.samples, rec.sample_rate)
    labeled = classify_beats(beats)
    summary = summarize(labeled, dog_weight_class=dog_weight_class)
    return write_report(
        labeled, summary, out_dir, samples=rec.samples, sample_rate=rec.sample_rate
    )
