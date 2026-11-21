"""
train.py
--------
Trains the Artificial Neural Network (scikit-learn MLPClassifier) for fake job
posting detection, runs a few small experiments and saves every artefact the
Streamlit app and the report need.

Run with:   python train.py

Outputs (inside model/):
    fake_job_mlp.joblib    -> the fitted pipeline (preprocessing + MLP)
    metadata.json          -> dataset summary + best experiment + metrics
    experiments.csv        -> the comparison table of all experiments
    figures/*.png          -> all graphs
"""

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # save figures to file instead of opening windows

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

import preprocessing as pp

FIG_DIR = pp.MODEL_DIR / "figures"
MODEL_PATH = pp.MODEL_DIR / "fake_job_mlp.joblib"
METADATA_PATH = pp.MODEL_DIR / "metadata.json"
EXPERIMENTS_PATH = pp.MODEL_DIR / "experiments.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Baseline network: 2 hidden layers (64 -> 32) + ReLU + Adam + lr 0.001
BASE_MLP_PARAMS = dict(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=100,
    random_state=RANDOM_STATE,
)

# A *small* set of meaningful experiments (one change at a time).
EXPERIMENTS = [
    ("Baseline (64-32, relu, lr=0.001)", {}),
    ("Learning rate = 0.0001", {"learning_rate_init": 0.0001}),
    ("Learning rate = 0.01", {"learning_rate_init": 0.01}),
    ("One hidden layer (32)", {"hidden_layer_sizes": (32,)}),
    ("Three hidden layers (128-64-32)", {"hidden_layer_sizes": (128, 64, 32)}),
    ("Fewer iterations (max_iter=20)", {"max_iter": 20}),
    ("Fewer iterations (max_iter=50)", {"max_iter": 50}),
    ("More iterations (max_iter=200)", {"max_iter": 200}),
    ("Activation = tanh", {"activation": "tanh"}),
    ("Activation = logistic", {"activation": "logistic"}),
]


# ------------------------------------------------------------------ utils ----
def oversample_minority(df: pd.DataFrame, target: str = pp.TARGET) -> pd.DataFrame:
    """
    Fix the class imbalance by RANDOM OVERSAMPLING of the training rows only.

    Only 4.8% of the postings are fraudulent. If we train on that ratio an
    MLP simply predicts "real" for everything and still looks 95% accurate.
    So we duplicate random fraudulent rows until the two classes are equal.
    Duplicates are only made inside the training split -> the test set stays
    untouched and keeps the real 4.8% ratio.
    """
    real = df[df[target] == 0]
    fake = df[df[target] == 1]
    fake_upsampled = fake.sample(n=len(real), replace=True, random_state=RANDOM_STATE)
    balanced = pd.concat([real, fake_upsampled])
    return balanced.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)


def build_model(mlp_params: dict) -> Pipeline:
    """Preprocessing + MLP in a single pipeline object (fit only on train data)."""
    return Pipeline(
        steps=[
            ("preprocess", pp.build_preprocessing_pipeline()),
            ("mlp", MLPClassifier(**mlp_params)),
        ]
    )


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Compute the metrics we report (accuracy is NOT enough here)."""
    y_pred = model.predict(X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "y_pred": y_pred,
    }


# ------------------------------------------------------------------- main ----
def main() -> None:
    pp.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- 1. load ----
    print("=" * 70)
    print("STEP 1 - Load and inspect the dataset")
    print("=" * 70)
    raw = pp.load_dataset()
    print(f"Shape            : {raw.shape[0]} rows x {raw.shape[1]} columns")
    print(f"Columns          : {list(raw.columns)}")
    print(f"Duplicate rows   : {raw.duplicated().sum()}")
    print("\nMissing values per column:")
    print(raw.isna().sum().to_string())

    target_counts = raw[pp.TARGET].value_counts()
    print("\nTarget column 'fraudulent':")
    for label, count in target_counts.items():
        name = "Fraudulent" if label == 1 else "Real"
        print(f"  {label} ({name:10s}) : {count:6d}  ({count / len(raw):.2%})")

    # ------------------------------------------------ 2. prepare data ----
    print("\n" + "=" * 70)
    print("STEP 2 - Clean, add simple features, select columns")
    print("=" * 70)
    df = pp.prepare_dataframe(raw)
    print(f"Rows after dropping duplicates : {len(df)}")
    print(f"Selected feature columns       : {pp.FEATURE_COLUMNS}")
    print(f"Number of input features       : {len(pp.FEATURE_COLUMNS)}")

    X = df.drop(columns=[pp.TARGET])
    y = df[pp.TARGET]

    # Stratified split keeps the 4.8% fraud ratio in both parts.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"\nTrain rows: {len(X_train)}  |  Test rows: {len(X_test)}")
    print(f"Fraud in train: {int(y_train.sum())} / {len(y_train)}")
    print(f"Fraud in test : {int(y_test.sum())} / {len(y_test)}")

    # Oversampling on the training split ONLY (no leakage into the test set).
    train_df = X_train.copy()
    train_df[pp.TARGET] = y_train.values
    train_balanced = oversample_minority(train_df)
    X_train_bal = train_balanced.drop(columns=[pp.TARGET])
    y_train_bal = train_balanced[pp.TARGET]
    print(
        "\nAfter oversampling the training set: "
        f"{len(X_train_bal)} rows "
        f"({int(y_train_bal.sum())} fraudulent / "
        f"{int((y_train_bal == 0).sum())} real)"
    )

    # ------------------------------------------------ 3. class balance ----
    print("\n" + "=" * 70)
    print("STEP 3 - Save figures for the report")
    print("=" * 70)
    plt.figure(figsize=(5, 3.5))
    ax = sns.barplot(
        x=["Real (0)", "Fraudulent (1)"],
        y=target_counts.values,
        hue=["Real (0)", "Fraudulent (1)"],
        palette=["#4C72B0", "#C44E52"],
        legend=False,
    )
    for i, v in enumerate(target_counts.values):
        ax.text(i, v + 200, f"{v} ({v / len(raw):.1%})", ha="center")
    ax.set_title("Class distribution in the dataset (imbalanced)")
    ax.set_ylabel("Number of postings")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "01_class_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 01_class_distribution.png")

    # --------------------------------------------- 4. train baseline ----
    print("\n" + "=" * 70)
    print("STEP 4 - Train the baseline MLP")
    print("=" * 70)
    print(f"Architecture: input({len(pp.FEATURE_COLUMNS)}) -> 64 -> 32 -> 1")
    start = time.time()
    model = build_model(BASE_MLP_PARAMS)
    model.fit(X_train_bal, y_train_bal)
    train_seconds = time.time() - start
    mlp = model.named_steps["mlp"]
    print(f"Training finished in {train_seconds:.1f}s after {mlp.n_iter_} iterations")

    res = evaluate(model, X_test, y_test)
    print("\n--- Test set performance (baseline) ---")
    print(f"Accuracy : {res['accuracy']:.4f}")
    print(f"Precision: {res['precision']:.4f}")
    print(f"Recall   : {res['recall']:.4f}")
    print(f"F1-score : {res['f1']:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, res["y_pred"], target_names=["Real", "Fraudulent"], zero_division=0))

    cm = confusion_matrix(y_test, res["y_pred"])
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)

    # ----------------------------------------------- 5. experiments ----
    print("\n" + "=" * 70)
    print("STEP 5 - Experiments (one change at a time)")
    print("=" * 70)
    rows = []
    for name, overrides in EXPERIMENTS:
        params = {**BASE_MLP_PARAMS, **overrides}
        t0 = time.time()
        exp_model = build_model(params)
        exp_model.fit(X_train_bal, y_train_bal)
        exp_res = evaluate(exp_model, X_test, y_test)
        rows.append(
            {
                "Experiment": name,
                "Hidden layers": str(params["hidden_layer_sizes"]),
                "Activation": params["activation"],
                "Learning rate": params["learning_rate_init"],
                "Max iterations": params["max_iter"],
                "Iterations run": exp_model.named_steps["mlp"].n_iter_,
                "Accuracy": round(exp_res["accuracy"], 4),
                "Precision": round(exp_res["precision"], 4),
                "Recall": round(exp_res["recall"], 4),
                "F1": round(exp_res["f1"], 4),
                "Train time (s)": round(time.time() - t0, 1),
            }
        )
        print(
            f"  {name:45s} acc={exp_res['accuracy']:.3f} "
            f"prec={exp_res['precision']:.3f} rec={exp_res['recall']:.3f} "
            f"f1={exp_res['f1']:.3f}"
        )

    exp_df = pd.DataFrame(rows)
    exp_df.to_csv(EXPERIMENTS_PATH, index=False)
    print(f"\nExperiment table saved to {EXPERIMENTS_PATH}")

    best_row = exp_df.loc[exp_df["F1"].idxmax()]
    print(f"Best experiment by F1: {best_row['Experiment']} (F1={best_row['F1']})")

    # ------------------------------------------------- 6. graphs ----
    print("\n" + "=" * 70)
    print("STEP 6 - Graphs")
    print("=" * 70)

    # (a) Training loss per iteration (MLPClassifier stores loss_curve_).
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(mlp.loss_curve_) + 1), mlp.loss_curve_, color="#4C72B0")
    plt.xlabel("Iteration (epoch)")
    plt.ylabel("Training loss")
    plt.title("Training loss vs iterations (baseline model)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "02_training_loss.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 02_training_loss.png")

    # (b) Test F1 vs number of training iterations. Only experiments where
    # max_iter is the single change are used (architecture stays the baseline).
    iter_df = exp_df[
        (exp_df["Hidden layers"] == str(BASE_MLP_PARAMS["hidden_layer_sizes"]))
        & (exp_df["Activation"] == BASE_MLP_PARAMS["activation"])
        & (exp_df["Learning rate"] == BASE_MLP_PARAMS["learning_rate_init"])
    ].sort_values("Max iterations")
    plt.figure(figsize=(6, 4))
    plt.plot(iter_df["Max iterations"], iter_df["F1"], marker="o", color="#55A868")
    for _, r in iter_df.iterrows():
        plt.annotate(f"{r['F1']:.3f}", (r["Max iterations"], r["F1"]),
                     textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    plt.xlabel("Max iterations (training epochs)")
    plt.ylabel("Test F1-score (fraudulent class)")
    plt.title("Model performance vs training iterations")
    plt.grid(alpha=0.3)
    plt.ylim(iter_df["F1"].min() - 0.06, iter_df["F1"].max() + 0.06)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "03_f1_vs_iterations.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 03_f1_vs_iterations.png")

    # (c) Experiment comparison.
    metrics_long = exp_df.melt(
        id_vars=["Experiment"],
        value_vars=["Precision", "Recall", "F1"],
        var_name="Metric",
        value_name="Score",
    )
    plt.figure(figsize=(11, 5))
    sns.barplot(data=metrics_long, x="Experiment", y="Score", hue="Metric")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.ylim(0, 1)
    plt.title("Comparison of MLP experiments on the test set")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "04_experiment_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 04_experiment_comparison.png")

    # (d) Confusion matrix of the baseline model.
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted real", "Predicted fraud"],
        yticklabels=["Actual real", "Actual fraud"],
        cbar=False,
    )
    plt.title("Confusion matrix (baseline MLP)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "05_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved 05_confusion_matrix.png")

    # ------------------------------------------------- 7. save model ----
    print("\n" + "=" * 70)
    print("STEP 7 - Save model and metadata")
    print("=" * 70)
    joblib.dump(model, MODEL_PATH)

    prep_fitted = model.named_steps["preprocess"]
    ohe = prep_fitted.named_transformers_["categorical"].named_steps["onehot"]
    cat_feature_names = ohe.get_feature_names_out(pp.CATEGORICAL_COLUMNS)

    metadata = {
        "dataset": {
            "rows": int(raw.shape[0]),
            "columns": int(raw.shape[1]),
            "duplicates_removed": int(raw.shape[0] - len(df)),
            "fraud_count": int(raw[pp.TARGET].sum()),
            "fraud_percent": round(float(raw[pp.TARGET].mean() * 100), 2),
        },
        "features": {
            "selected_columns": pp.FEATURE_COLUMNS,
            "categorical": pp.CATEGORICAL_COLUMNS,
            "numeric": pp.NUMERIC_COLUMNS,
            "n_features_after_encoding": int(cat_feature_names.shape[0] + len(pp.NUMERIC_COLUMNS)),
        },
        "split": {
            "train_rows": int(len(X_train)),
            "test_rows": int(len(X_test)),
            "train_rows_after_oversampling": int(len(X_train_bal)),
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
        },
        "baseline": {
            "params": {k: (list(v) if isinstance(v, tuple) else v) for k, v in BASE_MLP_PARAMS.items()},
            "iterations": int(mlp.n_iter_),
            "train_seconds": round(train_seconds, 1),
            "accuracy": round(res["accuracy"], 4),
            "precision": round(res["precision"], 4),
            "recall": round(res["recall"], 4),
            "f1": round(res["f1"], 4),
            "confusion_matrix": cm.tolist(),
            "classification_report": classification_report(
                y_test, res["y_pred"], target_names=["Real", "Fraudulent"], zero_division=0
            ),
        },
        "best_experiment": {
            "name": str(best_row["Experiment"]),
            "f1": float(best_row["F1"]),
        },
        "experiments": rows,
        # Used by the Streamlit dropdowns so the user picks a known category.
        "dropdown_options": {
            column: sorted(df[column].dropna().unique().tolist())
            for column in ["employment_type", "required_experience", "required_education"]
        },
        "top_categories": {
            column: df[column].value_counts().head(20).index.tolist()
            for column in ["industry", "function"]
        },
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"  model    -> {MODEL_PATH}")
    print(f"  metadata -> {METADATA_PATH}")
    print("\nDone. Start the demo with:  streamlit run app.py")


if __name__ == "__main__":
    main()