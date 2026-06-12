"""
model/train.py
─────────────────────────────────────────────────────────────────────────────
Production training pipeline.  Run once to generate all artifacts.

Usage:
    python -m model.train
    # OR from project root:
    python model/train.py

WHY EACH STEP EXISTS:
  1. Feature engineering  — must be identical to inference (see preprocess.py)
  2. Correlation drop     — removes redundant features that hurt generalisation
  3. ColumnTransformer    — StandardScaler for numerics, OHE for categoricals
                            Saved as preprocess.joblib for identical inference
  4. Class-balanced models— dataset has imbalanced job roles; balanced weights
                            prevent the model from collapsing to majority class
  5. OOF Stacking         — reduces overfitting vs. simple model averaging;
                            each base model is evaluated on held-out folds
  6. Artifacts saved      — feature_order.json is the single source of truth
                            for column alignment at inference time

COMMON FAILURE PREVENTION:
  • "Model always predicts same class" → class_weight="balanced" on all models
  • "Preprocessing mismatch"           → saved preprocess.joblib; loaded at inference
  • "Feature order mismatch"           → feature_order.json; reindex at inference
─────────────────────────────────────────────────────────────────────────────
"""

import json
import time
import warnings
import datetime
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    log_loss, top_k_accuracy_score, classification_report,
)
from sklearn.base import clone
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

from model.preprocess import add_features

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────────────────────────
DATA_PATH  = Path("career_dataset_final.csv")
LABEL_COL  = "job_role"
DOMAIN_COL = "career_domain"
SEED       = 42
OUT_DIR    = Path("artifacts_layer")
OUT_DIR.mkdir(exist_ok=True)

ROLE_MERGE_MAP = {
    "Cloud Engineer":               "Cloud Engineer & Platform/SRE",
    "Platform Engineer":            "Cloud Engineer & Platform/SRE",
    "Site Reliability Engineer":    "Cloud Engineer & Platform/SRE",
    "AI Engineer":                  "Data Scientist / ML / AI Engineer",
    "Data Scientist":               "Data Scientist / ML / AI Engineer",
    "Machine Learning Engineer":    "Data Scientist / ML / AI Engineer",
    "BI Analyst":                   "Data Scientist / ML / AI Engineer",
    "Analytics Engineer":           "Data & Analytics Engineer",
    "Data Engineer":                "Data & Analytics Engineer",
    "SOC Analyst":                  "Security Operations Analyst",
    "Security Analyst":             "Security Operations Analyst",
    "Security Operations Engineer": "Security Operations Analyst",
    "Cybersecurity Engineer":       "Cybersecurity & Cloud Security Engineer",
    "Cloud Security Engineer":      "Cybersecurity & Cloud Security Engineer",
    "QA Engineer":                  "QA & Automation Test Engineer",
    "Automation Test Engineer":     "QA & Automation Test Engineer",
}


def _aligned_proba(model, X, n_classes):
    p = model.predict_proba(X)
    if hasattr(model, "classes_"):
        a = np.zeros((p.shape[0], n_classes))
        a[:, model.classes_] = p
        return a
    return p


def find_high_corr_drops(df: pd.DataFrame, threshold: float = 0.85) -> list:
    """
    WHY: Highly correlated features give duplicate information but add noise
    and slow training.  We keep the one with higher variance.
    """
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        return []
    corr  = df[num_cols].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    drop  = set()
    for col in upper.columns:
        for partner in upper.index[upper[col] > threshold].tolist():
            drop.add(col if df[col].std() < df[partner].std() else partner)
    return list(drop)


def build_oof_stack_features(base_models, X_tr, y_tr, X_v, n_classes, n_splits=5):
    """
    WHY OOF (Out-Of-Fold) stacking:
      If base models see their own training data when generating meta-features,
      the meta-model over-trains on those over-fitted predictions.
      OOF ensures each prediction in meta_train is on held-out data.
    """
    kf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    n_base  = len(base_models)
    oof_tr  = np.zeros((X_tr.shape[0], n_base * n_classes))
    oof_v   = np.zeros((X_v.shape[0],  n_base * n_classes))
    for fold, (ti, vi) in enumerate(kf.split(X_tr, y_tr)):
        logger.info("    Fold %d/%d", fold + 1, n_splits)
        for j, (name, mdl) in enumerate(base_models.items()):
            m = clone(mdl)
            m.fit(X_tr[ti], y_tr[ti])
            oof_tr[vi, j*n_classes:(j+1)*n_classes] = _aligned_proba(m, X_tr[vi], n_classes)
            oof_v[:,  j*n_classes:(j+1)*n_classes] += _aligned_proba(m, X_v,     n_classes) / n_splits
    return oof_tr, oof_v


def run_training():
    sep = "=" * 60
    logger.info("%s\n  CAREER PREDICTION — TRAINING PIPELINE\n%s", sep, sep)

    # ── Load ──────────────────────────────────────────────────────────────────
    logger.info("\n[1] Loading data from %s", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    logger.info("    Shape: %s", df.shape)

    # Role→Domain mapping (before merging)
    role_to_domain = (
        df[[LABEL_COL, DOMAIN_COL]].dropna().drop_duplicates()
        .groupby(LABEL_COL)[DOMAIN_COL]
        .agg(lambda x: x.mode().iloc[0])
        .to_dict()
    )

    # ── Split ─────────────────────────────────────────────────────────────────
    logger.info("\n[2] Train / Val / Test split")
    y_raw = df[LABEL_COL].astype(str).replace(ROLE_MERGE_MAP)
    X_raw = df.drop(columns=[LABEL_COL, DOMAIN_COL], errors="ignore")

    X_tv, X_test, y_tv, y_test = train_test_split(
        X_raw, y_raw, test_size=0.20, random_state=SEED, stratify=y_raw
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tv, y_tv, test_size=0.125, random_state=SEED, stratify=y_tv   # 10% of total
    )
    logger.info("    Train=%d  Val=%d  Test=%d", len(X_train), len(X_val), len(X_test))

    # ── Feature Engineering ───────────────────────────────────────────────────
    logger.info("\n[3] Feature engineering")
    X_train_fe = add_features(X_train)
    X_val_fe   = add_features(X_val)
    X_test_fe  = add_features(X_test)
    logger.info("    Features after engineering: %d", X_train_fe.shape[1])

    # ── Correlation drop ──────────────────────────────────────────────────────
    drop_corr = find_high_corr_drops(X_train_fe, threshold=0.85)
    logger.info("    Dropping %d correlated features", len(drop_corr))
    X_train_sel = X_train_fe.drop(columns=drop_corr, errors="ignore")
    X_val_sel   = X_val_fe.drop(columns=drop_corr, errors="ignore")
    X_test_sel  = X_test_fe.drop(columns=drop_corr, errors="ignore")

    feature_order = X_train_sel.columns.tolist()

    # ── Preprocessing ─────────────────────────────────────────────────────────
    logger.info("\n[4] Building preprocessing pipeline")
    num_cols = [c for c in feature_order if pd.api.types.is_numeric_dtype(X_train_sel[c])]
    cat_cols = [c for c in feature_order if c not in num_cols]

    preprocessor = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("scl", StandardScaler())]),          num_cols),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown="ignore"))]), cat_cols),
    ], remainder="drop")

    X_tr_p = preprocessor.fit_transform(X_train_sel)
    X_v_p  = preprocessor.transform(X_val_sel)
    X_te_p = preprocessor.transform(X_test_sel)
    logger.info("    Processed shape: %s", X_tr_p.shape)

    # ── Label encoding ────────────────────────────────────────────────────────
    logger.info("\n[5] Label encoding — %d unique roles", y_train.nunique())
    le = LabelEncoder()
    y_tr_e = le.fit_transform(y_train)
    y_v_e  = le.transform(y_val)
    y_te_e = le.transform(y_test)
    n_cls  = len(le.classes_)

    # ── Base model training ───────────────────────────────────────────────────
    logger.info("\n[6] Training base models (class_weight=balanced)")

    lgbm_m = lgb.LGBMClassifier(
        n_estimators=800, learning_rate=0.07, num_leaves=80,
        min_child_samples=25, subsample=0.85, colsample_bytree=0.85,
        reg_alpha=0.1, reg_lambda=1.0, class_weight="balanced",
        random_state=SEED, verbosity=-1, n_jobs=-1,
    )
    xgb_m = XGBClassifier(
        n_estimators=800, learning_rate=0.07, max_depth=5, min_child_weight=4,
        subsample=0.75, colsample_bytree=0.75, gamma=0.2,
        reg_alpha=0.2, reg_lambda=3.0, eval_metric="mlogloss",
        use_label_encoder=False, random_state=SEED, n_jobs=-1, verbosity=0,
    )
    cat_m = CatBoostClassifier(
        iterations=600, learning_rate=0.07, depth=6, l2_leaf_reg=4.0,
        auto_class_weights="Balanced", od_type="Iter", od_wait=40,
        task_type="CPU", random_seed=SEED, verbose=0,
    )
    lr_m = LogisticRegression(
        C=0.8, penalty="l2", solver="lbfgs", max_iter=1000,
        class_weight="balanced", random_state=SEED
    )

    base_models = {"lgbm": lgbm_m, "xgb": xgb_m, "cat": cat_m, "lr": lr_m}
    fitted = {}
    for name, m in base_models.items():
        t0 = time.time()
        m.fit(X_tr_p, y_tr_e)
        preds = m.predict(X_v_p)
        acc   = accuracy_score(y_v_e, preds)
        f1    = f1_score(y_v_e, preds, average="macro")
        logger.info("    %-6s  val_acc=%.4f  val_f1=%.4f  [%.1fs]",
                    name.upper(), acc, f1, time.time() - t0)
        fitted[name] = m

    # ── OOF Stacking ──────────────────────────────────────────────────────────
    logger.info("\n[7] Building OOF stacking ensemble (5-fold)")
    t0 = time.time()
    meta_tr, meta_v = build_oof_stack_features(fitted, X_tr_p, y_tr_e, X_v_p, n_cls)

    meta_model = LogisticRegression(
        C=1.0, solver="lbfgs", penalty="l2",
        class_weight="balanced", max_iter=1000, random_state=SEED
    )
    meta_model.fit(meta_tr, y_tr_e)
    logger.info("    Stacking done in %.1fs", time.time() - t0)

    val_prob  = meta_model.predict_proba(meta_v)
    val_pred  = meta_model.predict(meta_v)
    val_top3  = float(top_k_accuracy_score(y_v_e, val_prob, k=3, labels=np.arange(n_cls)))
    val_f1    = float(f1_score(y_v_e, val_pred, average="macro"))
    val_acc   = float(accuracy_score(y_v_e, val_pred))
    logger.info("    STACKING VAL → acc=%.4f  f1=%.4f  top3=%.4f", val_acc, val_f1, val_top3)

    # ── Test evaluation ───────────────────────────────────────────────────────
    logger.info("\n[8] Test evaluation")
    parts_te = [_aligned_proba(m, X_te_p, n_cls) for m in fitted.values()]
    meta_te  = np.hstack(parts_te)
    te_prob  = meta_model.predict_proba(meta_te)
    te_pred  = meta_model.predict(meta_te)

    test_metrics = {
        "accuracy":  float(accuracy_score(y_te_e, te_pred)),
        "macro_f1":  float(f1_score(y_te_e, te_pred, average="macro")),
        "top3_acc":  float(top_k_accuracy_score(y_te_e, te_prob, k=3, labels=np.arange(n_cls))),
        "log_loss":  float(log_loss(y_te_e, te_prob, labels=np.arange(n_cls))),
    }
    for k, v in test_metrics.items():
        logger.info("    %-12s = %.4f", k, v)

    # ── Save artifacts ────────────────────────────────────────────────────────
    logger.info("\n[9] Saving artifacts to %s/", OUT_DIR)
    ensemble_bundle = {"base_models": fitted, "meta_model": meta_model}
    joblib.dump(ensemble_bundle, OUT_DIR / "stacking_ensemble.joblib")
    joblib.dump(preprocessor,   OUT_DIR / "preprocess.joblib")
    joblib.dump(le,             OUT_DIR / "label_encoder.joblib")

    # Merge role_to_domain with merged keys
    r2d_merged = {}
    for old_role, domain in role_to_domain.items():
        new_role = ROLE_MERGE_MAP.get(old_role, old_role)
        r2d_merged[new_role] = domain

    (OUT_DIR / "feature_order.json").write_text(json.dumps(feature_order, indent=2))
    (OUT_DIR / "dropped_correlated_features.json").write_text(
        json.dumps({"threshold": 0.85, "dropped_features": drop_corr}, indent=2)
    )
    (OUT_DIR / "role_to_domain.json").write_text(json.dumps(r2d_merged, indent=2))
    (OUT_DIR / "role_merge_map.json").write_text(json.dumps(ROLE_MERGE_MAP, indent=2))
    (OUT_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
    (OUT_DIR / "model_meta.json").write_text(json.dumps({
        "version":       "2.0.0",
        "trained_at":    datetime.datetime.now().isoformat(),
        "n_classes":     int(n_cls),
        "base_models":   list(fitted.keys()),
        "meta_model":    "LogisticRegression",
        "stacking_type": "OOF-5fold",
        "test_accuracy": round(test_metrics["accuracy"], 6),
        "test_macro_f1": round(test_metrics["macro_f1"], 6),
        "top3_acc":      round(test_metrics["top3_acc"], 6),
    }, indent=2))

    # Classification report
    report = classification_report(y_te_e, te_pred,
                                    target_names=le.classes_, zero_division=0)
    (OUT_DIR / "ensemble_classification_report_test.txt").write_text(report)

    logger.info("%s\n  DONE — Test Acc=%.4f  F1=%.4f  Top3=%.4f\n%s",
                sep, test_metrics["accuracy"], test_metrics["macro_f1"],
                test_metrics["top3_acc"], sep)


if __name__ == "__main__":
    run_training()
