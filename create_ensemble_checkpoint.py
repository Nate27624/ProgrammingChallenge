"""
create_ensemble_checkpoint.py
=============================
Bundle multiple trained runs into one ensemble checkpoint file:
  results/<team_name>/best_model.pt

The bundled checkpoint is compatible with test.py (checkpoint_type=ensemble_v1).

Why this exists:
  - The project expects a single directory under `results/<team_name>/`.
  - We still want ensemble performance, so this script packs multiple model
    checkpoints into one `best_model.pt` container.
  - `test.py` detects this container and runs averaged-logit inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_list(raw: str) -> list[str]:
    """Parse comma-separated run names, trimming whitespace and empties."""
    return [x.strip() for x in raw.split(",") if x.strip()]


def main() -> None:
    """Create one inference-ready ensemble checkpoint from multiple run folders."""
    p = argparse.ArgumentParser(description="Create a single-file ensemble checkpoint.")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--run_names", type=str, required=True, help="Comma-separated run names under results/")
    p.add_argument("--team_name", type=str, required=True, help="Output bundle directory name.")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    run_names = parse_list(args.run_names)
    if not run_names:
        raise ValueError("No run_names provided.")

    # `members` will be serialized directly inside the ensemble checkpoint.
    members = []
    base_sig = None
    base_stats = None
    for run_name in run_names:
        run_dir = results_dir / run_name
        model_path = run_dir / "best_model.pt"
        stats_path = run_dir / "norm_stats.pt"
        # Every member run must have both artifacts:
        #   - best_model.pt   (weights)
        #   - norm_stats.pt   (feature preprocessing metadata)
        if not model_path.exists() or not stats_path.exists():
            raise FileNotFoundError(f"Missing best_model.pt/norm_stats.pt for run: {run_name}")

        stats = torch.load(stats_path, map_location="cpu")
        # All members must share feature extraction settings so a single
        # test-time DataLoader/normalization is valid for the whole ensemble.
        sig = (
            stats.get("feature_type", "mel"),
            int(stats.get("n_features", 64)),
            bool(stats.get("include_deltas", False)),
        )
        if base_sig is None:
            # First model defines the canonical input signature.
            base_sig = sig
            base_stats = stats
        elif sig != base_sig:
            # Mixed feature signatures would invalidate ensemble inference
            # because one test dataloader cannot satisfy all models.
            raise ValueError(
                f"Incompatible feature signature for {run_name}: {sig} != {base_sig}"
            )

        # Member payload keeps enough info for `test.py` to reconstruct each
        # model class and instantiate it before loading its state dict.
        member = {
            "run_name": run_name,
            "model_name": stats.get("model_name", "baseline"),
            "model_config": stats.get("model_config", {}),
            "state_dict": torch.load(model_path, map_location="cpu"),
        }
        members.append(member)
        print(f"Added member: {run_name}")

    out_dir = results_dir / args.team_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # We store full member state dicts in a single artifact so test.py can
    # load and run ensemble inference from one --team_name directory.
    ensemble_ckpt = {
        "checkpoint_type": "ensemble_v1",
        "members": members,
    }
    torch.save(ensemble_ckpt, out_dir / "best_model.pt")

    # norm_stats for an ensemble is still required because test.py uses it to
    # build features (MFCC/Mel, deltas, normalization) before any model forward.
    norm_stats = {
        "mean": base_stats["mean"],
        "std": base_stats["std"],
        "include_deltas": base_sig[2],
        "feature_type": base_sig[0],
        "n_features": base_sig[1],
        "model_name": "ensemble",
        "model_config": {
            "ensemble_members": [m["run_name"] for m in members],
        },
    }
    torch.save(norm_stats, out_dir / "norm_stats.pt")
    (out_dir / "ensemble_members.json").write_text(
        json.dumps({"members": [m["run_name"] for m in members]}, indent=2),
        encoding="utf-8",
    )

    print(f"Saved ensemble best_model.pt to: {out_dir / 'best_model.pt'}")
    print(f"Saved ensemble norm_stats.pt to: {out_dir / 'norm_stats.pt'}")


if __name__ == "__main__":
    main()
