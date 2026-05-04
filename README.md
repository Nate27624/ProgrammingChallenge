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

## CV + Final Ensemble (Single `best_model.pt`)

This repo supports training multiple CV fold models across seeds and packaging them
into a single ensemble checkpoint file that `test.py` can load directly.

Run CV over seeds, select top fold models by validation F1, and build one final model:

```bash
python final_cv_ensemble_pipeline.py \
  --results_dir results \
  --base_name final_cv_ens \
  --seeds 42,2025 \
  --num_folds 3 \
  --top_k_models 4 \
  --ensemble_team_name final_ensemble_model \
  --wandb_mode offline \
  --train_args "--data_dir dataset --model_name bilstm_attention --feature_type mfcc --n_features 64 --include_deltas --specaugment --specaugment_on_gpu --cnn_channels 80 --hidden_size 192 --num_layers 1 --dropout 0.20106125299308247 --learning_rate 0.0008102814421521665 --weight_decay 2.392649157903331e-05 --max_grad_norm 0.0 --mixup_alpha 0.23642915524169064 --mixup_prob 0.4064569475358951 --loss_type ce --class_weighting --label_smoothing 0.03359513436382346 --scheduler_type plateau --warmup_epochs 3 --min_lr_ratio 0.1607512542884864 --batch_size 32 --num_time_masks 1 --time_mask_param 24 --num_freq_masks 2 --freq_mask_param 10 --speed_perturb_prob 0.0 --num_workers 0 --feature_cache_dir results/_feature_cache --patience 16 --patience_lr 6 --num_epochs 100"
```

This writes:

```text
results/final_ensemble_model/best_model.pt
results/final_ensemble_model/norm_stats.pt
```

Then evaluate normally:

```bash
python test.py --test_dir dataset/test --results_dir results --team_name final_ensemble_model --run_name final_ensemble_eval
```

## Root-level fallback behavior

`test.py` will use root `best_model.pt` and `norm_stats.pt` if
`results/<team_name>/best_model.pt` or `norm_stats.pt` are missing.
