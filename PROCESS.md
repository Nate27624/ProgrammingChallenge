# SER Training Process (Living Document)

This document tracks the full process used to build and tune our Speech Emotion Recognition (SER) system for the CSE 5526 programming challenge.

It is intended to be updated continuously as new experiments are run.

## 1. Goal and Constraints

- Primary goal: maximize weighted F1 on validation/public test.
- Stretch target: `> 0.70` weighted F1.
- Constraint: **no pretrained models** (so improvements must come from architecture/training/data recipe, not external pretrained encoders).

## 2. Starting Point

- Baseline model family started around LSTM-based pipeline.
- Baseline public weighted F1 (earlier run): ~`0.4276`.

## 3. Phase 1: Data + Features

Work implemented:

- SpecAugment support in the data pipeline.
- MFCC feature path (and/or mel path retained as option).
- Delta + delta-delta channel stacking (`include_deltas`) to form multi-channel time-frequency input.

Result:

- Stronger and more robust input features than plain baseline spectrogram setup.

## 4. Phase 2: Architecture Upgrade to Mamba

We pivoted from baseline recurrent setup to a CNN + bidirectional Mamba architecture.

Implemented model variants:

- `frontend_type`:
  - `basic_cnn`
  - `residual_cnn`
- `fusion_type`:
  - `concat`
  - `gated`
- `pooling_type`:
  - `meanmax`
  - `attention`

Rationale:

- CNN front-end extracts local time-frequency patterns.
- Bidirectional Mamba stack models sequence context with lower overhead than transformer-style alternatives.
- Fusion/pooling knobs let us tune representation quality and regularization behavior.

## 5. Phase 3: Structured Optuna Search

### 5.1 Why structured instead of free search

Observed issue: free search can waste trials on weak/unstable configurations.

Solution: two-stage process.

### 5.2 Stage A (Architecture Screen)

- Purpose: isolate architecture shape effects with fixed capacity/regularization.
- Search space: 8 architecture combinations from:
  - `frontend_type` x `fusion_type` x `pooling_type`
- Repeats used for stability (seed repeat dimension).

Latest complete Stage A study:

- Study: `mamba_stage_a_safe`
- Trials: 16 total (`15 COMPLETE`, `1 FAIL`)
- Best single val F1: `0.5722`
- Top-performing family overall: `basic_cnn` + `gated` variants.

Important takeaway:

- Architecture differences were real, but close at the top; variance mattered.

### 5.3 Stage B (Focused Coupled Search)

- Purpose: optimize capacity + regularization jointly within shortlisted architecture pool.
- Key tuned knobs included:
  - Mamba/CNN capacity (`d_model`, `cnn_channels`, `d_state`, layers, etc.)
  - regularization (`dropout`, `weight_decay`, `label_smoothing`)
  - augmentation strength (`num_time_masks`, `time_mask_param`, `num_freq_masks`, `freq_mask_param`)
  - learning rate + batch size

Latest main Stage B study:

- Study: `mamba_stage_b_main`
- Trials: 74 total
  - `29 COMPLETE`
  - `44 PRUNED`
  - `1 FAIL`
- Best single-trial val F1: `0.65339` (trial 52)

Important pattern:

- Many top trials converged to similar architecture/capacity regime:
  - `basic_cnn:gated:meanmax`
  - `mamba_d_model=128`
  - `cnn_channels=64`
  - `mamba_expand=4`
  - `batch_size=32`

## 6. Bias/Variance Selection Layer

Because single-trial winners were noisy, we added a post-Optuna reranking step:

- Script: `stage_b_bias_variance.py`
- Method:
  - take top `K` Stage B trials
  - rerun each across fixed seeds
  - score with `mean_val_f1 - lambda_std * std_val_f1`

Latest reranking outcome (from `mamba_stage_b_main`):

- Best robust config: trial 53
- Stage-B trial value: `0.65096`
- Multi-seed mean: `0.65440`
- Std: `0.00150`
- Score (`lambda_std=0.5`): `0.65365`

Key lesson:

- Highest single trial was not always the most reliable model.
- Robust mean/std selection is required for trustworthy model choice.

## 7. Stability/Infrastructure Issues Encountered

### 7.1 Windows pagefile + dataloader worker issues

- Error seen: `WinError 1455` (paging file too small).
- Mitigation:
  - use `num_workers=0`
  - lower batch size for stability
  - increase Windows virtual memory/pagefile.

### 7.2 CUDA OOM

Mitigations added:

- AMP (mixed precision) support in search loops.
- Trial-level OOM catch -> prune trial instead of crashing study.
- Explicit CUDA cache cleanup between trials.

## 8. Most Recent Upgrade Block (for >0.70 push)

Implemented in latest commit (`caf804c` on `Nate`):

- `train.py`:
  - new loss options:
    - `loss_type`: `ce` or `focal`
    - `focal_gamma`
    - `class_weighting` toggle
  - new scheduler options:
    - `scheduler_type`: `plateau`, `warmup_invsqrt`, `warmup_cosine`
    - `warmup_epochs`
    - `min_lr_ratio`
  - stores these settings in saved model metadata.

- `optuna_search.py`:
  - Stage B now tunes:
    - `loss_type`, `focal_gamma`, `class_weighting`
    - `scheduler_type`, `warmup_epochs`, `min_lr_ratio`
  - retrain command output includes new knobs.

- `stage_b_bias_variance.py`:
  - now supports reranking configs that use new loss/scheduler settings.
  - retrain command output includes new knobs.

## 9. Current Recommended Run Sequence

### Step 1: Focused Stage B recipe search

```bash
python optuna_search.py \
  --data_dir dataset \
  --results_dir results \
  --study_name mamba_stage_b_recipe \
  --mode stage_b \
  --stage_b_architectures basic_cnn:gated:meanmax \
  --n_trials 50 \
  --max_epochs 35 \
  --early_stop_patience 8 \
  --patience_lr 3 \
  --num_workers 0 \
  --timeout 43200
```

### Step 2: Bias/variance reranking

```bash
python stage_b_bias_variance.py \
  --data_dir dataset \
  --results_dir results \
  --study_name mamba_stage_b_recipe \
  --top_k 5 \
  --seeds 42,1337 \
  --lambda_std 0.5 \
  --max_epochs 30 \
  --early_stop_patience 6 \
  --patience_lr 3 \
  --num_workers 0
```

### Step 3: Retrain selected config

- Use generated:
  - `results/optuna/mamba_stage_b_recipe/bias_variance_retrain_command.txt`

## 10. Interpreting Results Going Forward

When comparing runs:

- Prefer robust score:
  - `mean_val_f1 - 0.5 * std`
- If two configs are within ~`0.01` mean F1:
  - treat as tie
  - retrain both and consider ensembling.

Expected trajectory (no pretraining):

- Current robust zone: mid `0.65x`.
- Next gains should come from improved generalization recipe (warmup + loss + weighting + augmentation coupling), not just longer training.

## 11. Update Protocol for This File

For each major run, append:

- date/time
- command used
- study name
- trial counts and statuses
- best single-trial F1
- best robust mean/std score
- decision taken for next step

---

Last updated: 2026-04-30

## 12. Latest Run Snapshot (m5-1 / 2026-05-01)

Study inspected: `mamba_stage_b_recipe`

Observed state from study DB:

- `28` total trials
- `13 COMPLETE`, `14 PRUNED`, `1 FAIL`
- best trial: `#11`, val weighted F1 = `0.666876`

Top trial characteristics so far:

- architecture fixed to: `basic_cnn:gated:meanmax`
- best configs strongly favored:
  - `loss_type=ce`
  - `scheduler_type=plateau`
  - `class_weighting=False`
  - batch mostly `32`
  - learning rate near high end (`~8.5e-4` to `1.0e-3`)
  - moderate dropout (`~0.29` to `0.34`)
  - label smoothing around `0.04` to `0.06`

Interpretation:

- This run is materially better than prior robust `~0.654` range.
- New warmup/focal/weighting knobs did not dominate early; CE+plateau currently leads.
- Since best is at trial 11 and no later trial has surpassed it, immediate next step is robust reranking (mean/std) instead of extending broad random search.

Immediate next step:

1. Run `stage_b_bias_variance.py` on `mamba_stage_b_recipe` top trials.
2. Retrain top 2 robust configs.
3. Ensemble top 2 if close.
