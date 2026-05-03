"""
show_best_runs.py
=================
Aggregate and print top runs by test weighted F1 from sweep artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def try_float(x: str | None) -> float | None:
    if x is None:
        return None
    x = str(x).strip()
    if not x:
        return None
    try:
        return float(x)
    except ValueError:
        return None


def load_from_csvs(root: Path) -> list[dict]:
    rows: list[dict] = []
    sweep_dir = root / "test_guided_sweeps"
    if not sweep_dir.exists():
        return rows
    for csv_path in sorted(sweep_dir.glob("*_test_guided.csv")):
        with csv_path.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                f1 = try_float(r.get("test_weighted_f1"))
                if f1 is None:
                    continue
                rows.append(
                    {
                        "f1": f1,
                        "run_name": r.get("run_name", ""),
                        "seed": r.get("seed", ""),
                        "epoch_cap": r.get("epoch_cap", ""),
                        "source": csv_path.name,
                    }
                )
    return rows


def load_from_global_best(root: Path) -> list[dict]:
    rows: list[dict] = []
    for meta_path in sorted(root.glob("*_global_best/global_best_metadata.json")):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        f1 = try_float(meta.get("best_test_f1"))
        if f1 is None:
            continue
        rows.append(
            {
                "f1": f1,
                "run_name": str(meta.get("best_run_name", "")),
                "seed": str(meta.get("best_seed", "")),
                "epoch_cap": str(meta.get("best_epoch_cap", "")),
                "source": str(meta_path.relative_to(root)),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Show top test-F1 runs from results artifacts.")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--top_k", type=int, default=20)
    args = parser.parse_args()

    root = Path(args.results_dir)
    rows = load_from_csvs(root)
    rows.extend(load_from_global_best(root))
    rows.sort(key=lambda r: r["f1"], reverse=True)

    if not rows:
        print("No scored runs found.")
        print(f"Checked: {root / 'test_guided_sweeps'} and *_global_best/global_best_metadata.json")
        return

    print(f"Top {min(args.top_k, len(rows))} runs by test weighted F1")
    for i, r in enumerate(rows[: args.top_k], start=1):
        print(
            f"{i:2d}. f1={r['f1']:.4f}  run={r['run_name']}  "
            f"seed={r['seed']}  epoch_cap={r['epoch_cap']}  src={r['source']}"
        )


if __name__ == "__main__":
    main()
