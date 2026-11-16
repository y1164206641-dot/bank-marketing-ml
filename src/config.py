"""Central configuration for the bank-marketing churn/response project."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
FIGURE_DIR = REPORT_DIR / "figures"

DATA_FILE = RAW_DIR / "bank-additional" / "bank-additional-full.csv"
RAW_ZIP_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

for _d in (RAW_DIR, PROCESSED_DIR, ARTIFACT_DIR, REPORT_DIR, FIGURE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- data contract -------------------------------------------------------
TARGET = "y"
POSITIVE = "yes"
LEAKY_FEATURES = ["duration"]  # known only after the call ends
TEST_FRACTION = 0.20

# Primary evaluation uses a stratified random holdout: the campaign's base rate
# swings from ~3% (2008) to ~46% (late 2010), so a pure time split turns the
# holdout into a label-shift stress test rather than a like-for-like benchmark.
# The temporal split is still run, as an explicit robustness experiment.
SPLIT_STRATEGY = "random"
RANDOM_STATE = 42

# --- business assumptions (EUR) -----------------------------------------
# Illustrative numbers for the campaign simulator; all are configurable and
# a sensitivity grid is produced by evaluate.profit_sensitivity().
VALUE_PER_CONVERSION = 80.0  # net margin of one term-deposit subscription
COST_PER_CONTACT = 5.0  # agent time + call cost
CAPACITY_FRACTION = 0.20  # share of the list a call centre can work through
