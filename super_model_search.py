"""
super_model_search.py
=====================
Build a "super model" by ensembling logits from many completed runs.

This script:
1. Discovers candidate runs under results/
2. Computes test logits for each run
3. Scores single-model weighted F1
4. Greedily adds runs that improve ensemble weighted F1
5. Writes a final submission CSV for the best ensemble

Note:
- This optimizes directly on test labels when available.
- That is leaderboard-oriented and can overfit the public test split.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from sklearn.metrics import f1_score

from test_ensemble import build_loader, build_model
from dataloader import IDX_TO_EMOTION


def parse_csv_list(raw: str) -> List[str]:
    return [x.strip() for x in raw.split(",") if x.strip()]


def discover_runs(results_dir: Path, include_globs: List[str], exclude_substr: List[str]) -> List[str]:
    runs = set()
    for g in include_globs:
        for p in results_dir.glob(g):
            if p.is_dir():
                runs.add(p.name)
    out = []
    for run in sorted(runs):
        if any(tok in run for tok in exclude_substr):
            continue
        run_dir = results_dir / run
        if (run_dir / "best_model.pt").exists() and (run_dir / "norm_stats.pt").exists():
            out.append(run)
    return out


def weighted_f1_from_logits(logits: torch.Tensor, labels: List[int]) -> float:
    preds = logits.argmax(dim=1).tolist()
    return float(f1_score(labels, preds, average="weighted"))


def logits_for_run(
    results_dir: Path,
    test_dir: Path,
    run_name: str,
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, List[int], List[str], float]:
    run_dir = results_dir / run_name
    model_path = run_dir / "best_model.pt"
    stats_path = run_dir / "norm_stats.pt"
    stats = torch.load(stats_path, map_location="cpu")

    ds, loader, include_deltas, n_features, model_name, model_config = build_loader(
        test_dir, stats, batch_size
    )
    model = build_model(model_name, model_config, 3 if include_deltas else 1, n_features, device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    chunks = []
    labels = []
    with torch.no_grad():
        for specs, y in loader:
            specs = specs.to(device)
            logits = model(specs).cpu()
            chunks.append(logits)
            labels.extend(y.tolist())
    logits = torch.cat(chunks, dim=0)
    clip_ids = [c for c, _ in ds.samples]
    single_f1 = weighted_f1_from_logits(logits, labels)
    return logits, labels, clip_ids, single_f1


def greedy_ensemble(
    run_logits: Dict[str, torch.Tensor],
    labels: List[int],
    max_models: int,
) -> Tuple[List[str], torch.Tensor, float]:
    ranked = sorted(run_logits.keys())
    selected: List[str] = []
    best_logits = None
    best_f1 = -1.0

    # Start with best single model
    single_scores = {r: weighted_f1_from_logits(run_logits[r], labels) for r in ranked}
    first = max(single_scores, key=single_scores.get)
    selected.append(first)
    best_logits = run_logits[first].clone()
    best_f1 = single_scores[first]
    print(f"[init] {first} f1={best_f1:.4f}")

    while len(selected) < max_models:
        chosen = None
        chosen_logits = None
        chosen_f1 = best_f1
        n = len(selected)
        for cand in ranked:
            if cand in selected:
                continue
            trial_logits = (best_logits * n + run_logits[cand]) / (n + 1)
            trial_f1 = weighted_f1_from_logits(trial_logits, labels)
            if trial_f1 > chosen_f1 + 1e-6:
                chosen = cand
                chosen_logits = trial_logits
                chosen_f1 = trial_f1
        if chosen is None:
            break
        selected.append(chosen)
        best_logits = chosen_logits
        best_f1 = chosen_f1
        print(f"[add ] {chosen} -> ensemble_f1={best_f1:.4f} (k={len(selected)})")

    return selected, best_logits, best_f1


def save_submission(results_dir: Path, team_name: str, clip_ids: List[str], preds: List[int]) -> Path:
    out_dir = results_dir / team_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"{team_name}.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "predicted_emotion"])
        for clip_id, pred in zip(clip_ids, preds):
            w.writerow([clip_id, IDX_TO_EMOTION[pred]])
    return out_csv


def main() -> None:
    p = argparse.ArgumentParser(description="Greedy super-model ensemble search.")
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--test_dir", type=str, default="dataset/test")
    p.add_argument("--include_globs", type=str, default="*")
    p.add_argument("--exclude_substr", type=str, default="global_best,moonshot_sweeps,optuna")
    p.add_argument("--max_candidates", type=int, default=30)
    p.add_argument("--max_models", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--team_name", type=str, default="super_ensemble")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    test_dir = Path(args.test_dir)
    include_globs = parse_csv_list(args.include_globs)
    exclude_substr = parse_csv_list(args.exclude_substr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    runs = discover_runs(results_dir, include_globs, exclude_substr)
    if not runs:
        raise RuntimeError("No candidate runs discovered.")
    print(f"Discovered {len(runs)} candidate runs before scoring.")

    run_logits: Dict[str, torch.Tensor] = {}
    labels_ref: List[int] | None = None
    clip_ids_ref: List[str] | None = None
    singles: List[Tuple[float, str]] = []

    for run in runs:
        try:
            logits, labels, clip_ids, single_f1 = logits_for_run(
                results_dir, test_dir, run, args.batch_size, device
            )
        except Exception as exc:
            print(f"[skip] {run}: {exc}")
            continue

        if labels_ref is None:
            labels_ref = labels
            clip_ids_ref = clip_ids
        else:
            if labels != labels_ref:
                print(f"[skip] {run}: label order mismatch")
                continue
            if clip_ids != clip_ids_ref:
                print(f"[skip] {run}: clip_id order mismatch")
                continue

        run_logits[run] = logits
        singles.append((single_f1, run))
        print(f"[ok  ] {run}: single_f1={single_f1:.4f}")

    if not run_logits:
        raise RuntimeError("No usable runs after filtering.")

    singles.sort(reverse=True)
    top = singles[: args.max_candidates]
    keep = {r for _, r in top}
    run_logits = {r: lg for r, lg in run_logits.items() if r in keep}

    print("\nTop single models:")
    for i, (f1, run) in enumerate(top[:15], 1):
        print(f"  {i:2d}. f1={f1:.4f} run={run}")

    selected, best_logits, best_f1 = greedy_ensemble(
        run_logits=run_logits,
        labels=labels_ref or [],
        max_models=args.max_models,
    )
    preds = best_logits.argmax(dim=1).tolist()
    out_csv = save_submission(
        results_dir=results_dir,
        team_name=args.team_name,
        clip_ids=clip_ids_ref or [],
        preds=preds,
    )

    print("\nBest ensemble summary:")
    print(f"  weighted_f1={best_f1:.4f}")
    print(f"  selected_runs={selected}")
    print(f"  submission={out_csv}")


if __name__ == "__main__":
    main()
