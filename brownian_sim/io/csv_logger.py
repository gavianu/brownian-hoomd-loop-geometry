"""CSV loggers: tranziții între piese + contori per piesă."""
from __future__ import annotations

import csv
import os
from typing import Any

import numpy as np


class CSVLoggers:
    """Agregator pentru CSV-urile standard ale unei simulări.

    Fișiere:
      - transitions.csv   step, particle_id, from, to
      - piece_counts.csv  step, <name1>, <name2>, ...
    """

    def __init__(self, out_dir: str, piece_names: list[str]) -> None:
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.piece_names = piece_names

        self.trans_path = os.path.join(out_dir, "transitions.csv")
        with open(self.trans_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "particle_id", "from", "to"])

        self.counts_path = os.path.join(out_dir, "piece_counts.csv")
        with open(self.counts_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step"] + piece_names)

    def write_frame(self, step, positions, velocities, piece_idx, assembly):
        counts = [int((piece_idx == k).sum()) for k in range(len(self.piece_names))]
        with open(self.counts_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([int(step)] + counts)

    def write_transitions(self, step, ids, prev_idx, new_idx, names):
        with open(self.trans_path, "a", newline="") as f:
            w = csv.writer(f)
            for i in ids:
                fr = names[prev_idx[i]] if prev_idx[i] >= 0 else "OUT"
                to = names[new_idx[i]] if new_idx[i] >= 0 else "OUT"
                w.writerow([int(step), int(i), fr, to])
