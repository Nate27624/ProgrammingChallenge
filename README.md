# Final SER Training + Submission

## Included files

- `dataloader.py`
- `model.py`
- `train.py`
- `test.py`
- `run_cv.py`
- `create_ensemble_checkpoint.py`
- `final_cv_ensemble_pipeline.py`
- `distill_train.py`
- `baseline.py`, `wandb_compat.py` (runtime dependencies)

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

## Recommended Training (Single Run)

This is the recommended direct training command for the current best
`bilstm_attention` setup.

```bash
python train.py \
  --data_dir dataset \
  --results_dir results \
  --team_name final_run \
  --run_name final_run \
  --model_name bilstm_attention \
  --feature_type mfcc \
  --n_features 64 \
  --include_deltas \
  --specaugment \
  --specaugment_on_gpu \
  --cnn_channels 80 \
  --hidden_size 192 \
  --num_layers 1 \
  --dropout 0.20 \
  --learning_rate 8.1e-4 \
  --weight_decay 2.39e-5 \
  --max_grad_norm 1.0 \
  --mixup_alpha 0.236 \
  --mixup_prob 0.406 \
  --loss_type ce \
  --class_weighting \
  --label_smoothing 0.0336 \
  --scheduler_type plateau \
  --warmup_epochs 3 \
  --min_lr_ratio 0.16 \
  --batch_size 32 \
  --num_epochs 120 \
  --patience 16 \
  --patience_lr 6 \
  --num_time_masks 1 \
  --time_mask_param 24 \
  --num_freq_masks 2 \
  --freq_mask_param 10 \
  --num_workers 2 \
  --feature_cache_dir results/_feature_cache \
  --seed 2026
```

This writes artifacts under `results/final_run/`.

## Recommended CV + Ensemble Flow

Train multiple folds with `run_cv.py`, then bundle top models into one
`best_model.pt` ensemble.

```bash
python run_cv.py \
  --base_team_name final_cv_s2026 \
  --num_folds 3 \
  --seed 2026 \
  --results_dir results \
  --wandb_mode offline \
  --train_args "--data_dir dataset --model_name bilstm_attention --feature_type mfcc --n_features 64 --include_deltas --specaugment --specaugment_on_gpu --cnn_channels 80 --hidden_size 192 --num_layers 1 --dropout 0.20 --learning_rate 8.1e-4 --weight_decay 2.39e-5 --max_grad_norm 1.0 --mixup_alpha 0.236 --mixup_prob 0.406 --loss_type ce --class_weighting --label_smoothing 0.0336 --scheduler_type plateau --warmup_epochs 3 --min_lr_ratio 0.16 --batch_size 32 --num_epochs 120 --patience 16 --patience_lr 6 --num_time_masks 1 --time_mask_param 24 --num_freq_masks 2 --freq_mask_param 10 --num_workers 2 --feature_cache_dir results/_feature_cache"
```

```bash
python create_ensemble_checkpoint.py \
  --results_dir results \
  --run_names "final_cv_s2026_fold0,final_cv_s2026_fold1,final_cv_s2026_fold2" \
  --team_name final_ensemble
```

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

## Optional: Distillation

Use `distill_train.py` to train a single student model from one or more
teacher runs in `results/`.

```bash
python distill_train.py \
  --data_dir dataset \
  --results_dir results \
  --teacher_runs "runA,runB" \
  --team_name distilled_student \
  --student_model bilstm_attention
```
