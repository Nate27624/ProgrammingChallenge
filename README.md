# Final SER Submission Branch

This branch is trimmed to submission-focused files only.

## Included files

- `dataloader.py`
- `model.py`
- `train.py`
- `test.py`
- `best_model.pt`
- `norm_stats.pt`
- `final_submission.csv`
- `baseline.py`, `mamba_minimal.py`, `wandb_compat.py` (runtime dependencies)

## Environment

Install required packages:

```bash
pip install torch torchaudio scikit-learn pandas matplotlib wandb tqdm
```

Expected dataset layout:

```text
dataset/
  train/
    audio/
    train_labels.csv
  test/
    audio/
    test_labels.csv
```

## Train

```bash
python train.py \
  --data_dir dataset \
  --results_dir results \
  --team_name final_run \
  --run_name final_run
```

This writes artifacts under `results/final_run/`.

## Test / Generate CSV

```bash
python test.py \
  --test_dir dataset/test \
  --results_dir results \
  --team_name final_run \
  --run_name final_eval
```

Output CSV will be:

```text
results/final_run/final_run.csv
```

## Root-level fallback behavior

`test.py` will use root `best_model.pt` and `norm_stats.pt` if
`results/<team_name>/best_model.pt` or `norm_stats.pt` are missing.

