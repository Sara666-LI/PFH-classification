#!/usr/bin/env python3
"""End-to-end PFH binary classification pipeline.

Supports:
- Clinical, Radiomics, Foundation, and fusion models.
- LR/SVM/RF candidate classifiers.
- SMOTE within cross-validation folds.
- Cox-style (univariate p-value) + bootstrap feature selection.
- SHAP explainability.
- ROC, radar, boxplot, and scatter visualizations.
- Fold-level train/validation probability exports.
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import statsmodels.api as sm
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    balanced_accuracy_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")


@dataclass
class FoldResult:
    model_name: str
    classifier: str
    repeat: int
    fold: int
    split: str
    auc: float
    balanced_accuracy: float
    sensitivity: float
    specificity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PFH binary classification pipeline")
    parser.add_argument("--radiomics-file", default="df_radiomics_aligned.xlsx")
    parser.add_argument("--foundation-file", default="df_found_aligned.xlsx")
    parser.add_argument("--target-col", default="PFH")
    parser.add_argument("--id-col", default="ID")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--n-repeats", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--bootstrap-iters", type=int, default=200)
    parser.add_argument("--max-selected-features", type=int, default=30)
    return parser.parse_args()


def align_datasets(df_rad: pd.DataFrame, df_found: pd.DataFrame, id_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if id_col in df_rad.columns and id_col in df_found.columns:
        common_ids = sorted(set(df_rad[id_col]).intersection(df_found[id_col]))
        df_rad = df_rad[df_rad[id_col].isin(common_ids)].copy().sort_values(id_col).reset_index(drop=True)
        df_found = df_found[df_found[id_col].isin(common_ids)].copy().sort_values(id_col).reset_index(drop=True)
    else:
        n = min(len(df_rad), len(df_found))
        df_rad = df_rad.iloc[:n].copy().reset_index(drop=True)
        df_found = df_found.iloc[:n].copy().reset_index(drop=True)
    return df_rad, df_found


def cox_style_feature_selection(X: pd.DataFrame, y: pd.Series, max_features: int = 30, p_thresh: float = 0.05) -> List[str]:
    selected = []
    pvals = {}
    y_num = pd.to_numeric(y, errors="coerce")

    for col in X.columns:
        x_col = pd.to_numeric(X[col], errors="coerce")
        if x_col.nunique(dropna=True) < 2:
            continue
        try:
            x_const = sm.add_constant(x_col.fillna(x_col.median()))
            model = sm.Logit(y_num, x_const).fit(disp=0)
            pval = model.pvalues.get(col, 1.0)
            pvals[col] = pval
            if pval < p_thresh:
                selected.append(col)
        except Exception:
            continue

    if not selected:
        selected = sorted(pvals, key=pvals.get)[:max_features]
    else:
        selected = sorted(selected, key=lambda c: pvals[c])[:max_features]
    return selected


def bootstrap_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    n_bootstrap: int = 200,
    random_state: int = 42,
    top_k_each_bootstrap: int = 20,
    max_features: int = 30,
) -> List[str]:
    rng = np.random.default_rng(random_state)
    counts = {c: 0 for c in X.columns}
    X_num = X.apply(pd.to_numeric, errors="coerce").fillna(X.median(numeric_only=True))

    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(X_num), len(X_num))
        X_bs = X_num.iloc[idx]
        y_bs = y.iloc[idx]
        try:
            rf = RandomForestClassifier(n_estimators=300, random_state=random_state, class_weight="balanced")
            rf.fit(X_bs, y_bs)
            imps = pd.Series(rf.feature_importances_, index=X_num.columns).sort_values(ascending=False)
            for col in imps.head(min(top_k_each_bootstrap, len(imps))).index:
                counts[col] += 1
        except Exception:
            continue

    ranked = pd.Series(counts).sort_values(ascending=False)
    selected = ranked[ranked > 0].head(max_features).index.tolist()
    return selected if selected else X.columns[: min(max_features, X.shape[1])].tolist()


def select_rad_found_features(X: pd.DataFrame, y: pd.Series, args: argparse.Namespace) -> List[str]:
    cox_sel = cox_style_feature_selection(X, y, max_features=args.max_selected_features)
    boot_sel = bootstrap_feature_selection(
        X,
        y,
        n_bootstrap=args.bootstrap_iters,
        random_state=args.random_state,
        max_features=args.max_selected_features,
    )
    inter = sorted(set(cox_sel).intersection(boot_sel))
    if inter:
        return inter[: args.max_selected_features]
    merged = cox_sel + [f for f in boot_sel if f not in cox_sel]
    return merged[: args.max_selected_features]


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat_cols = [c for c in X.columns if c not in num_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols),
    ])


def classifier_zoo(random_state: int) -> Dict[str, object]:
    return {
        "LR": LogisticRegression(max_iter=3000, class_weight="balanced", random_state=random_state),
        "SVM": SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=random_state),
        "RF": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    y_pred = (y_prob >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "auc": roc_auc_score(y_true, y_prob),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def evaluate_modality(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    preprocessor = build_preprocessor(X)
    clf_dict = classifier_zoo(args.random_state)
    cv = RepeatedStratifiedKFold(
        n_splits=args.n_splits,
        n_repeats=args.n_repeats,
        random_state=args.random_state,
    )

    fold_rows = []
    pred_rows = []
    auc_by_clf = {k: [] for k in clf_dict}

    for clf_name, clf in clf_dict.items():
        for split_idx, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
            repeat = split_idx // args.n_splits
            fold = split_idx % args.n_splits

            X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
            y_train, y_valid = y.iloc[tr_idx], y.iloc[va_idx]

            pipe = ImbPipeline([
                ("preprocessor", clone(preprocessor)),
                ("smote", SMOTE(random_state=args.random_state)),
                ("clf", clone(clf)),
            ])
            pipe.fit(X_train, y_train)

            train_prob = pipe.predict_proba(X_train)[:, 1]
            valid_prob = pipe.predict_proba(X_valid)[:, 1]

            train_metrics = compute_metrics(y_train.to_numpy(), train_prob)
            valid_metrics = compute_metrics(y_valid.to_numpy(), valid_prob)
            auc_by_clf[clf_name].append(valid_metrics["auc"])

            fold_rows.append(
                {
                    "model": model_name,
                    "classifier": clf_name,
                    "repeat": repeat,
                    "fold": fold,
                    "split": "train",
                    **train_metrics,
                }
            )
            fold_rows.append(
                {
                    "model": model_name,
                    "classifier": clf_name,
                    "repeat": repeat,
                    "fold": fold,
                    "split": "valid",
                    **valid_metrics,
                }
            )

            for idx_local, idx_global in enumerate(tr_idx):
                pred_rows.append(
                    {
                        "model": model_name,
                        "classifier": clf_name,
                        "repeat": repeat,
                        "fold": fold,
                        "split": "train",
                        "row_index": int(idx_global),
                        "y_true": int(y.iloc[idx_global]),
                        "y_prob": float(train_prob[idx_local]),
                    }
                )
            for idx_local, idx_global in enumerate(va_idx):
                pred_rows.append(
                    {
                        "model": model_name,
                        "classifier": clf_name,
                        "repeat": repeat,
                        "fold": fold,
                        "split": "valid",
                        "row_index": int(idx_global),
                        "y_true": int(y.iloc[idx_global]),
                        "y_prob": float(valid_prob[idx_local]),
                    }
                )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(pred_rows)

    mean_auc = {k: np.mean(v) for k, v in auc_by_clf.items() if v}
    best_clf = max(mean_auc, key=mean_auc.get)

    fold_df.to_csv(output_dir / f"{model_name}_fold_metrics.csv", index=False)
    pred_df.to_csv(output_dir / f"{model_name}_fold_predictions.csv", index=False)

    return fold_df, pred_df, best_clf


def late_fusion(
    fusion_name: str,
    pred_frames: Dict[str, pd.DataFrame],
    args: argparse.Namespace,
    output_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    keys = list(pred_frames)
    merged = None
    for k in keys:
        tmp = pred_frames[k].copy()
        tmp = tmp.rename(columns={"y_prob": f"prob_{k}", "y_true": "y_true_tmp"})
        keep = ["classifier", "repeat", "fold", "split", "row_index", f"prob_{k}", "y_true_tmp"]
        tmp = tmp[keep]
        if merged is None:
            merged = tmp.rename(columns={"y_true_tmp": "y_true"})
        else:
            merged = merged.merge(
                tmp.drop(columns=["y_true_tmp"]),
                on=["classifier", "repeat", "fold", "split", "row_index"],
                how="inner",
            )

    prob_cols = [c for c in merged.columns if c.startswith("prob_")]
    merged["y_prob"] = merged[prob_cols].mean(axis=1)

    fold_rows = []
    for (clf, rep, fold, split), g in merged.groupby(["classifier", "repeat", "fold", "split"]):
        m = compute_metrics(g["y_true"].to_numpy(), g["y_prob"].to_numpy())
        fold_rows.append(
            {
                "model": fusion_name,
                "classifier": clf,
                "repeat": rep,
                "fold": fold,
                "split": split,
                **m,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    pred_df = merged[["classifier", "repeat", "fold", "split", "row_index", "y_true", "y_prob"]].copy()
    pred_df.insert(0, "model", fusion_name)

    fold_df.to_csv(output_dir / f"{fusion_name}_fold_metrics.csv", index=False)
    pred_df.to_csv(output_dir / f"{fusion_name}_fold_predictions.csv", index=False)
    return fold_df, pred_df


def plot_roc_curves(all_pred_df: pd.DataFrame, output_dir: Path) -> None:
    valid = all_pred_df[all_pred_df["split"] == "valid"].copy()
    for model_name, g in valid.groupby("model"):
        plt.figure(figsize=(6, 5))
        for clf_name, gc in g.groupby("classifier"):
            fpr, tpr, _ = roc_curve(gc["y_true"], gc["y_prob"])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{clf_name} (AUC={roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves - {model_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"roc_{model_name}.png", dpi=200)
        plt.close()

    plt.figure(figsize=(8, 6))
    for model_name, g in valid.groupby("model"):
        fpr, tpr, _ = roc_curve(g["y_true"], g["y_prob"])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Comparison Across All Models")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_all_models_comparison.png", dpi=220)
    plt.close()


def plot_metric_boxplots(all_fold_df: pd.DataFrame, output_dir: Path) -> None:
    valid = all_fold_df[all_fold_df["split"] == "valid"].copy()
    metrics = ["auc", "balanced_accuracy", "sensitivity", "specificity"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, metric in zip(axes.ravel(), metrics):
        sns.boxplot(data=valid, x="model", y=metric, ax=ax)
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "metrics_boxplots_all_models.png", dpi=220)
    plt.close()


def plot_radar(all_fold_df: pd.DataFrame, output_dir: Path) -> None:
    valid = all_fold_df[all_fold_df["split"] == "valid"].copy()
    metrics = ["auc", "balanced_accuracy", "sensitivity", "specificity"]
    summary = valid.groupby("model")[metrics].mean().reset_index()

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    plt.figure(figsize=(9, 8))
    ax = plt.subplot(111, polar=True)
    for _, row in summary.iterrows():
        values = row[metrics].tolist()
        values += values[:1]
        ax.plot(angles, values, label=row["model"])
        ax.fill(angles, values, alpha=0.08)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title("Radar Plot of Mean Validation Metrics")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / "radar_metrics.png", dpi=220)
    plt.close()


def make_scatter_risk_plot(all_pred_df: pd.DataFrame, output_dir: Path) -> None:
    valid = all_pred_df[all_pred_df["split"] == "valid"].copy()
    clinical = (
        valid[valid["model"] == "MClinic"]
        .groupby("row_index")
        .agg(clinical_risk=("y_prob", "mean"), label=("y_true", "first"))
    )
    rad_found = (
        valid[valid["model"] == "MRad-Found"]
        .groupby("row_index")
        .agg(rad_found_risk=("y_prob", "mean"))
    )
    scatter_df = clinical.join(rad_found, how="inner").reset_index()

    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        data=scatter_df,
        x="clinical_risk",
        y="rad_found_risk",
        hue="label",
        palette={0: "#1f77b4", 1: "#d62728"},
        alpha=0.8,
    )
    plt.title("Risk Score Scatter: Clinical vs Rad-Found")
    plt.xlabel("Clinical Risk Score")
    plt.ylabel("Rad-Found Risk Score")
    plt.tight_layout()
    plt.savefig(output_dir / "scatter_clinical_vs_rad_found_risk.png", dpi=220)
    plt.close()


def generate_shap(
    model_name: str,
    X: pd.DataFrame,
    y: pd.Series,
    best_clf_name: str,
    args: argparse.Namespace,
    output_dir: Path,
) -> None:
    pre = build_preprocessor(X)
    clf = classifier_zoo(args.random_state)[best_clf_name]
    pipe = ImbPipeline([
        ("preprocessor", pre),
        ("smote", SMOTE(random_state=args.random_state)),
        ("clf", clf),
    ])
    pipe.fit(X, y)

    X_t = pipe.named_steps["preprocessor"].transform(X)
    if hasattr(X_t, "toarray"):
        X_t = X_t.toarray()

    feature_names = pipe.named_steps["preprocessor"].get_feature_names_out().tolist()
    X_t_df = pd.DataFrame(X_t, columns=feature_names)

    clf_fit = pipe.named_steps["clf"]
    sample = X_t_df.sample(min(200, len(X_t_df)), random_state=args.random_state)

    if best_clf_name == "RF":
        explainer = shap.TreeExplainer(clf_fit)
        shap_values = explainer.shap_values(sample)
        vals = shap_values[1] if isinstance(shap_values, list) else shap_values
    elif best_clf_name == "LR":
        explainer = shap.LinearExplainer(clf_fit, sample)
        vals = explainer.shap_values(sample)
    else:
        background = sample.sample(min(100, len(sample)), random_state=args.random_state)
        explainer = shap.KernelExplainer(clf_fit.predict_proba, background)
        vals = explainer.shap_values(sample, nsamples=100)
        vals = vals[1] if isinstance(vals, list) else vals

    plt.figure()
    shap.summary_plot(vals, sample, show=False, max_display=20)
    plt.tight_layout()
    plt.savefig(output_dir / f"shap_summary_{model_name}.png", dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df_rad = pd.read_excel(args.radiomics_file)
    df_found = pd.read_excel(args.foundation_file)
    df_rad, df_found = align_datasets(df_rad, df_found, args.id_col)

    y = pd.to_numeric(df_rad[args.target_col], errors="coerce").fillna(0).astype(int)

    clinic_cols = [c for c in ["Sex", "Age", "AFP"] if c in df_rad.columns]
    clinic_use = [c for c in ["Age", "AFP"] if c in clinic_cols]
    if not clinic_use:
        clinic_use = clinic_cols

    non_clinical = [c for c in df_rad.columns if c not in clinic_cols + [args.target_col, args.id_col]]
    X_clinic = df_rad[clinic_use].copy()
    X_rad_raw = df_rad[non_clinical].copy()
    X_found_raw = df_found[[c for c in df_found.columns if c not in [args.target_col, args.id_col]]].copy()

    rad_features = select_rad_found_features(X_rad_raw, y, args)
    found_features = select_rad_found_features(X_found_raw, y, args)

    X_rad = X_rad_raw[rad_features].copy()
    X_found = X_found_raw[found_features].copy()

    model_data = {
        "MClinic": X_clinic,
        "MRad": X_rad,
        "MFound": X_found,
    }

    all_fold = []
    all_pred = []
    best_clfs = {}

    for name, X in model_data.items():
        fold_df, pred_df, best_clf = evaluate_modality(name, X, y, args, out)
        all_fold.append(fold_df)
        all_pred.append(pred_df)
        best_clfs[name] = best_clf
        generate_shap(name, X, y, best_clf, args, out)

    pred_lookup = {name: df for name, df in zip(model_data.keys(), all_pred)}

    # Feature-level fusion datasets
    fusion_feature_data = {
        "MClinic-Rad": pd.concat([X_clinic, X_rad], axis=1),
        "MRad-Found": pd.concat([X_rad, X_found], axis=1),
        "MClinic-Rad-Found": pd.concat([X_clinic, X_rad, X_found], axis=1),
    }
    for name, X in fusion_feature_data.items():
        fold_df, pred_df, best_clf = evaluate_modality(name, X, y, args, out)
        all_fold.append(fold_df)
        all_pred.append(pred_df)
        best_clfs[name] = best_clf
        generate_shap(name, X, y, best_clf, args, out)

    # Late fusion strategy using probability averaging
    lf_defs = {
        "LateFusion_MClinic-Rad": {"MClinic": pred_lookup["MClinic"], "MRad": pred_lookup["MRad"]},
        "LateFusion_MRad-Found": {"MRad": pred_lookup["MRad"], "MFound": pred_lookup["MFound"]},
        "LateFusion_MClinic-Rad-Found": {
            "MClinic": pred_lookup["MClinic"],
            "MRad": pred_lookup["MRad"],
            "MFound": pred_lookup["MFound"],
        },
    }
    for lf_name, frames in lf_defs.items():
        fold_df, pred_df = late_fusion(lf_name, frames, args, out)
        all_fold.append(fold_df)
        all_pred.append(pred_df)

    all_fold_df = pd.concat(all_fold, ignore_index=True)
    all_pred_df = pd.concat(all_pred, ignore_index=True)

    all_fold_df.to_csv(out / "all_models_fold_metrics.csv", index=False)
    all_pred_df.to_csv(out / "all_models_fold_predictions.csv", index=False)

    plot_roc_curves(all_pred_df, out)
    plot_metric_boxplots(all_fold_df, out)
    plot_radar(all_fold_df, out)
    make_scatter_risk_plot(all_pred_df, out)

    feature_manifest = {
        "clinic_features_used": clinic_use,
        "radiomics_selected_features": rad_features,
        "foundation_selected_features": found_features,
        "best_classifier_per_model": best_clfs,
    }
    (out / "feature_and_model_manifest.json").write_text(json.dumps(feature_manifest, indent=2))

    print("Pipeline completed successfully.")
    print(f"Outputs saved to: {out.resolve()}")


if __name__ == "__main__":
    main()
