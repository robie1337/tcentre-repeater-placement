"""Shared setup for the scripts: import path, output directories, logging."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def banner(title: str) -> None:
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def step(text: str) -> None:
    print(f"\n-- {text}")


def save_frame(frame, name: str):
    """Write a results table to CSV and report where it went."""
    path = RESULTS / name
    frame.to_csv(path, index=False)
    print(f"   wrote {path.relative_to(ROOT)}  ({len(frame)} rows)")
    return path
