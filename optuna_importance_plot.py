import argparse
import pandas as pd
import matplotlib.pyplot as plt
import optuna
from optuna.importance import get_param_importances

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db_path", required=True, help="Path to .db file")
    p.add_argument("--study_name", required=True, help="Optuna study name")
    p.add_argument("--out_png", default="param_importance.png")
    p.add_argument("--out_csv", default="param_importance.csv")
    args = p.parse_args()

    storage = f"sqlite:///{args.db_path}"
    study = optuna.load_study(study_name=args.study_name, storage=storage)

    imps = get_param_importances(study)  # dict: param -> importance
    df = pd.DataFrame({"param": list(imps.keys()), "importance": list(imps.values())})
    df = df.sort_values("importance", ascending=True)
    df.to_csv(args.out_csv, index=False)

    plt.figure(figsize=(9, 6))
    plt.barh(df["param"], df["importance"])
    plt.xlabel("Importance")
    plt.title(f"Optuna Hyperparameter Importance\n{args.study_name}")
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=200)
    print(f"Saved: {args.out_png}")
    print(f"Saved: {args.out_csv}")
    print(df.sort_values('importance', ascending=False).head(10).to_string(index=False))

if __name__ == "__main__":
    main()
