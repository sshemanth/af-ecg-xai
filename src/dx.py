"""Diagnosis codes of the Challenge 2021 datasets (Chapman-Shaoxing, Georgia):
code names, rhythm groups and reading every code from the .hea headers."""
from pathlib import Path

import pandas as pd

from src import config
from src.evaluation import wilson_ci
from src.labels import _parse_header

CODES = pd.read_csv(Path(__file__).resolve().parent / "snomed_codes.csv",
                    dtype={"code": str})
NAMES = dict(zip(CODES["code"], CODES["name"]))
ABBR = dict(zip(CODES["code"], CODES["abbr"]))

# first match wins; codes from the PhysioNet/CinC Challenge 2021 mapping
RHYTHM_GROUPS = [
    ("Paced", ["10370003", "251268003", "251266004"]),
    ("Supraventricular tachycardia (SVT, AT, AVNRT, AVRT, junctional)",
     ["426761007", "67198005", "713422000", "251166008", "233897008", "426648003"]),
    ("Wandering pacemaker / atrial rhythm",
     ["17366009", "195101003", "106068003", "233892002"]),
    ("Premature atrial beats", ["284470004", "63593006", "251173003", "251170000", "251168009"]),
    ("Premature ventricular beats", ["427172004", "17338001", "251182009"]),
    ("Sinus arrhythmia", ["427393009"]),
    ("Sinus tachycardia", ["427084000"]),
    ("Sinus bradycardia", ["426177001"]),
    ("Sinus rhythm", ["426783006"]),
]

# sinus-type rhythm codes: an ECG labelled AF that also carries one of these
# most likely had AF at another time, not in this 10 s strip
SINUS_CODES = {"426783006": "sinus rhythm", "426177001": "sinus bradycardia",
               "427084000": "sinus tachycardia", "427393009": "sinus arrhythmia"}


def main_rhythm(dx: set) -> str:
    for name, codes in RHYTHM_GROUPS:
        if dx & set(codes):
            return name
    return "Other / no rhythm code"


def read_dx(labels: pd.DataFrame, key: str) -> pd.Series:
    """Diagnosis codes per record, from each ECG's .hea header."""
    out = {}
    for rid, path in zip(labels["record_id"], labels["path"]):
        hea = Path(str(path) + ".hea")
        if not hea.exists():                     # data folder moved: search for it
            hits = list(config.CHALLENGE_DIRS[key].rglob(f"{rid}.hea"))
            hea = hits[0] if hits else None
        out[rid] = set(_parse_header(hea)["dx"]) if hea else set()
    return pd.Series(out, name="dx")


def rate(k: int, n: int) -> str:
    if n == 0:
        return "–"
    lo, hi = wilson_ci(k, n)
    return f"{k}/{n} = {k / n:.2f} ({lo:.2f}–{hi:.2f})"
