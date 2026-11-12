"""
Load the World of Warcraft Avatar History (WoWAH) data and cut it into sessions.

Raw data (Kaggle, "Warcraft Avatar History", CC0 Public Domain):
    data/raw/wowah_data.csv   10.8 million rows, one row = one avatar seen online
                              during one ~10-minute snapshot of the server in 2008
    data/raw/zones.csv        zone metadata (type: City, Zone, Dungeon, Battleground...)

Outputs (cached, git-ignored):
    data/processed/records.parquet   cleaned rows + zone type
    data/processed/sessions.parquet  one row per play session

A *session* = consecutive snapshots of the same avatar with less than
SESSION_GAP minutes between them. Play time of a session = (end - start) + one
snapshot interval (10 min), because each snapshot represents ~10 minutes of play.

Usage:  python src/prepare_data.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
RAW = PROJECT_DIR / "data" / "raw"
PROC = PROJECT_DIR / "data" / "processed"
SNAPSHOT_MIN = 10   # the crawler recorded the server population every ~10 minutes
SESSION_GAP = 30    # minutes without being seen -> new session
KAGGLE_URL = "https://www.kaggle.com/api/v1/datasets/download/mylesoneill/warcraft-avatar-history"


def ensure_raw_data() -> None:
    """Download and unzip the Kaggle dataset if the CSV is missing."""
    if (RAW / "wowah_data.csv").exists():
        return
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / "wowah.zip"
    print(f"Downloading {KAGGLE_URL} (~75 MB) ...")
    # curl is available on Windows 10+, macOS and Linux
    subprocess.run(["curl", "-sSL", "-o", str(zip_path), KAGGLE_URL], check=True)
    import zipfile
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW)
    zip_path.unlink()


def load_records() -> pd.DataFrame:
    df = pd.read_csv(RAW / "wowah_data.csv", skipinitialspace=True)
    df.columns = [c.strip() for c in df.columns]
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%m/%d/%y %H:%M:%S")

    # Zone -> zone type (City, Zone, Dungeon, Battleground, Arena, ...)
    zones = pd.read_csv(RAW / "zones.csv")
    zone_type = dict(zip(zones["Zone_Name"], zones["Type"]))
    # zones.csv labels these capital cities as "Zone"; in the game they are the
    # main social hubs (banks, auction house), so we count them as cities.
    for city in ["Shattrath City", "Silvermoon City", "Dalaran"]:
        zone_type[city] = "City"
    df["zone_type"] = df["zone"].map(zone_type)
    # One zone name is not in zones.csv (an arena with a non-ASCII suffix).
    df.loc[df["zone_type"].isna() & df["zone"].str.contains("Dalaran", na=False), "zone_type"] = "Arena"
    df["zone_type"] = df["zone_type"].fillna("Other")

    for col in ["race", "charclass", "zone", "zone_type"]:
        df[col] = df[col].astype("category")
    df["level"] = df["level"].astype("int16")
    df["guild"] = df["guild"].astype("int32")
    df["char"] = df["char"].astype("int32")
    return df.sort_values(["char", "timestamp"]).reset_index(drop=True)


def sessionize(df: pd.DataFrame) -> pd.DataFrame:
    """Group consecutive snapshots of the same avatar into sessions."""
    df = df.sort_values(["char", "timestamp"])
    gap = df["timestamp"].diff().dt.total_seconds().div(60)
    new = (df["char"] != df["char"].shift()) | (gap > SESSION_GAP)
    sid = new.cumsum()
    s = df.groupby(sid, sort=False).agg(
        char=("char", "first"), start=("timestamp", "first"), end=("timestamp", "last"),
        level_start=("level", "first"), level_end=("level", "last"),
        n_snapshots=("timestamp", "size"))
    s["minutes"] = (s["end"] - s["start"]).dt.total_seconds() / 60 + SNAPSHOT_MIN
    return s.reset_index(drop=True)


def load_all(force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    rec_path, ses_path = PROC / "records.parquet", PROC / "sessions.parquet"
    if rec_path.exists() and ses_path.exists() and not force:
        return pd.read_parquet(rec_path), pd.read_parquet(ses_path)
    ensure_raw_data()
    print("Reading wowah_data.csv (10.8 M rows) ...")
    rec = load_records()
    print("Building sessions ...")
    ses = sessionize(rec)
    PROC.mkdir(parents=True, exist_ok=True)
    rec.to_parquet(rec_path, index=False)
    ses.to_parquet(ses_path, index=False)
    return rec, ses


if __name__ == "__main__":
    rec, ses = load_all(force="--force" in sys.argv)
    same = rec["char"].eq(rec["char"].shift())
    gaps = rec["timestamp"].diff().dt.total_seconds().div(60)[same]
    print(f"{len(rec):,} records, {rec['char'].nunique():,} avatars, {len(ses):,} sessions")
    print(f"median gap between two snapshots of the same avatar: {gaps.median():.1f} min")
    print(f"median session length: {ses['minutes'].median():.0f} min")
    print(rec["zone_type"].value_counts(normalize=True).round(3))
