"""One-off script to fetch small local test fixtures from PhysioNet.

Run manually: .venv/bin/python scripts/fetch_fixtures.py

Requires network access. Fixtures are committed to the repo afterward so
the test suite never needs network access.
"""
import os
import wfdb

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures")


def fetch_mitbih_sample():
    """First 60 seconds of MIT-BIH record 119 (360Hz), with beat annotations.

    Record 100 (the originally planned record) was checked and rejected:
    it has only a single PVC in the entire 30-minute recording, occurring
    at ~25 minutes in, well outside any short fixture window - a 60s slice
    of it contains zero 'V' (PVC) beats, which defeats the purpose of a
    fixture meant to exercise PVC classification.

    Record 119 was used instead: it has frequent PVCs in a ventricular
    bigeminy pattern (alternating N/V beats), giving 19 'V' annotations in
    the first 60 seconds alone - a much more useful fixture for validating
    PVC detection/classification (see Task 7)."""
    out_dir = os.path.join(FIXTURES_DIR, "mitdb_119")
    os.makedirs(out_dir, exist_ok=True)
    record = wfdb.rdrecord("119", pn_dir="mitdb", sampto=21600)
    ann = wfdb.rdann("119", "atr", pn_dir="mitdb", sampto=21600)
    wfdb.wrsamp(
        record_name="119",
        fs=record.fs,
        units=record.units,
        sig_name=record.sig_name,
        p_signal=record.p_signal,
        write_dir=out_dir,
    )
    ann.record_name = "119"
    ann.wrann(write_dir=out_dir)
    print(f"Wrote MIT-BIH fixture to {out_dir}")


def fetch_physiozoo_dog_sample():
    """A short canine ECG sample from the PhysioZoo Mammalian NSR Database.

    Record confirmed via the PhysioNet file browser at
    https://physionet.org/files/physiozoo/1.0.0/wfdb_format/dog/ - the
    database stores each dog recording as Dog_NN/Dog_NN.{dat,hea,qrs} under
    wfdb_format/dog/. Dog_01 is used here (single-lead, 500Hz, normal sinus
    rhythm, ~134k samples / ~4.5 minutes). Annotations use extension "qrs"
    (not "atr" like MIT-BIH) and all beats are labeled "N" (normal) since
    this database is normal-sinus-rhythm-only.
    """
    RECORD_NAME = "Dog_01"
    PN_DIR = "physiozoo/1.0.0/wfdb_format/dog/Dog_01"
    out_dir = os.path.join(FIXTURES_DIR, "physiozoo_dog1")
    os.makedirs(out_dir, exist_ok=True)
    record = wfdb.rdrecord(RECORD_NAME, pn_dir=PN_DIR)
    ann = wfdb.rdann(RECORD_NAME, "qrs", pn_dir=PN_DIR)
    wfdb.wrsamp(
        record_name=RECORD_NAME,
        fs=record.fs,
        units=record.units,
        sig_name=record.sig_name,
        p_signal=record.p_signal,
        write_dir=out_dir,
    )
    ann.record_name = RECORD_NAME
    ann.wrann(write_dir=out_dir)
    print(f"Wrote PhysioZoo fixture to {out_dir}")


if __name__ == "__main__":
    fetch_mitbih_sample()
    fetch_physiozoo_dog_sample()
