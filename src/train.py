"""Train and compare models for machine-failure prediction (AI4I 2020).

Trains Logistic Regression, Random Forest and XGBoost with class-imbalance
handling, evaluates with PR-AUC / ROC-AUC / F1 (5-fold CV + holdout test),
saves the best pipeline to models/model.joblib and figures/metrics to reports/.

Run: python src/train.py
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from data import CATEGORICAL_FEATURES, NUMERIC_FEATURES, load_dataset

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
MODELS.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)
RANDOM_STATE = 42


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ]
    )


def main() -> None:
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    scale_pos_weight = neg / pos
    print(f"Train: {len(y_train):,} rows | failures: {pos} ({y_train.mean():.2%}) | scale_pos_weight={scale_pos_weight:.1f}")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, scale_pos_weight=scale_pos_weight,
            eval_metric="aucpr", random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {"roc_auc": "roc_auc", "pr_auc": "average_precision", "f1": "f1"}

    results: dict[str, dict] = {}
    pipelines: dict[str, Pipeline] = {}
    for name, clf in models.items():
        pipe = Pipeline([("prep", build_preprocessor()), ("clf", clf)])
        cvres = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pipe.fit(X_train, y_train)
        proba = pipe.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "cv_pr_auc": float(cvres["test_pr_auc"].mean()),
            "cv_roc_auc": float(cvres["test_roc_auc"].mean()),
            "cv_f1": float(cvres["test_f1"].mean()),
            "test_pr_auc": float(average_precision_score(y_test, proba)),
            "test_roc_auc": float(roc_auc_score(y_test, proba)),
            "test_f1": float(f1_score(y_test, pred)),
        }
        pipelines[name] = pipe
        r = results[name]
        print(f"  {name:<20} PR-AUC={r['test_pr_auc']:.3f}  ROC-AUC={r['test_roc_auc']:.3f}  F1={r['test_f1']:.3f}")

    best_name = max(results, key=lambda n: results[n]["test_pr_auc"])
    best = pipelines[best_name]
    print(f"\nBest model: {best_name}")

    # --- persist model + metrics ---
    joblib.dump(best, MODELS / "model.joblib")
    proba = best.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, pred)
    report = classification_report(y_test, pred, target_names=["No failure", "Failure"], output_dict=True)
    metrics = {
        "best_model": best_name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "positive_rate": float(y.mean()),
        "results": results,
        "best_confusion_matrix": cm.tolist(),
        "best_classification_report": report,
    }
    (REPORTS / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # --- model comparison bar chart ---
    comp = pd.DataFrame(results).T[["test_pr_auc", "test_roc_auc", "test_f1"]]
    ax = comp.plot.bar(rot=15, figsize=(8, 4.5), color=["#1f77b4", "#2ca02c", "#ff7f0e"])
    ax.set_title("Model comparison (held-out test set)")
    ax.set_ylabel("score"); ax.set_ylim(0, 1.05); ax.legend(["PR-AUC", "ROC-AUC", "F1"])
    plt.tight_layout(); plt.savefig(REPORTS / "model_comparison.png", dpi=120); plt.close()

    # --- ROC + PR curves for best ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    RocCurveDisplay.from_predictions(y_test, proba, ax=axes[0], name=best_name)
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3); axes[0].set_title("ROC curve")
    PrecisionRecallDisplay.from_predictions(y_test, proba, ax=axes[1], name=best_name)
    axes[1].set_title("Precision-Recall curve")
    plt.tight_layout(); plt.savefig(REPORTS / "roc_pr_curves.png", dpi=120); plt.close()

    # --- confusion matrix ---
    ConfusionMatrixDisplay(cm, display_labels=["No failure", "Failure"]).plot(cmap="Blues")
    plt.title(f"Confusion matrix — {best_name}")
    plt.tight_layout(); plt.savefig(REPORTS / "confusion_matrix.png", dpi=120); plt.close()

    # --- SHAP summary (tree models) ---
    try:
        prep = best.named_steps["prep"]
        clf = best.named_steps["clf"]
        feat_names = list(prep.get_feature_names_out())
        X_test_t = prep.transform(X_test)
        explainer = shap.TreeExplainer(clf)
        sv = explainer.shap_values(X_test_t)
        if isinstance(sv, list):
            sv = sv[1]
        elif getattr(sv, "ndim", 2) == 3:
            sv = sv[:, :, 1]
        shap.summary_plot(sv, X_test_t, feature_names=feat_names, show=False)
        plt.tight_layout(); plt.savefig(REPORTS / "shap_summary.png", dpi=120, bbox_inches="tight"); plt.close()
        print("Saved SHAP summary plot.")
    except Exception as exc:  # noqa: BLE001
        print(f"SHAP step skipped: {exc}")

    print(f"\nSaved model -> {MODELS/'model.joblib'}")
    print(f"Saved metrics + figures -> {REPORTS}")


if __name__ == "__main__":
    main()
