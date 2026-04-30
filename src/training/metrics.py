import csv
import json
import os
from argparse import Namespace
from datetime import datetime
from pathlib import Path
from typing import Sequence

from src.settings import settings


class MetricsLogger:

    def __init__(self, experiment: str, args: Namespace, fieldnames: Sequence[str], base_dir: Path | None = None):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._run_dir = (base_dir if base_dir is not None else settings.RESULTS_DIR / experiment) / timestamp
        self._run_dir.mkdir(parents=True, exist_ok=True)
        self._csv_path = self._run_dir / "metrics.csv"
        self._fieldnames = list(fieldnames)

        with open(self._run_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2, default=str)

    @property
    def run_dir(self) -> Path:
        return self._run_dir

    def log(self, metrics: dict) -> None:
        file_exists = self._csv_path.exists() and self._csv_path.stat().st_size > 0
        with open(self._csv_path, "a", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=self._fieldnames, extrasaction="ignore", restval=""
            )
            if not file_exists:
                writer.writeheader()
            writer.writerow(metrics)
            f.flush()
            os.fsync(f.fileno())
