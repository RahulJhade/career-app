"""
Career Prediction - Full Training Pipeline
Faithfully reproduces the notebook logic end-to-end.
Run this once to produce all artifacts in artifacts_layer/
"""

import json
import time
import warnings
import datetime
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
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    log_loss, top_k_accuracy_score, classification_report,
)
from sklearn.base import clone
import lightgbm as lgb
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# ── Constants ────────────────────────────────────────────────────────────────
DATA_PATH  = "career_dataset_final.csv"
LABEL_COL  = "job_role"
DOMAIN_COL = "career_domain"
SEED       = 42
OUT_DIR    = Path("artifacts_layer")
OUT_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("  CAREER PREDICTION — MODEL TRAINING PIPELINE")
print("=" * 60)

# ── LAYER 1 — Load Data ───────────────────────────────────────────────────────
print("\n[LAYER 1] Loading data...")
df = pd.read_csv(DATA_PATH)
print(f"  Dataset shape: {df.shape}")

role_to_domain = (
    df[[LABEL_COL, DOMAIN_COL]].dropna().drop_duplicates()
    .groupby(LABEL_COL)[DOMAIN_COL]
    .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
    .to_dict()
)
(OUT_DIR / "role_to_domain.json").write_text(json.dumps(role_to_domain, indent=2))

# ── LAYER 2 — Train/Val/Test Split ────────────────────────────────────────────
print("\n[LAYER 2] Splitting data...")
y = df[LABEL_COL].astype(str)
X = df.drop(columns=[LABEL_COL, DOMAIN_COL], errors="ignore").copy()

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)
val_relative = 0.10 / 0.80
X_train, X_val, y_train, y_val = train_test_split(
    X_trainval, y_trainval, test_size=val_relative, random_state=SEED, stratify=y_trainval
)
print(f"  Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

# ── LAYER 3 — Feature Engineering ────────────────────────────────────────────
print("\n[LAYER 3] Feature engineering...")

def add_advanced_engineered_features(X_in: pd.DataFrame) -> pd.DataFrame:
    X = X_in.copy()
    skill_interest_pairs = [
        ("programming_skill",       "interest_dev_overall",             "prog"),
        ("data_analytics_skill",    "interest_data_overall",            "data_analytics"),
        ("data_science_ml_skill",   "interest_data_overall",            "data_science"),
        ("cloud_devops_skill",      "interest_cloud_infra_overall",     "cloud"),
        ("cybersecurity_skill",     "interest_cybersecurity",           "cyber"),
        ("ui_ux_design_skill",      "interest_ui_ux_design",            "uiux"),
        ("business_analysis_skill", "interest_business_and_management", "biz"),
    ]
    for skill_col, interest_col, prefix in skill_interest_pairs:
        if skill_col in X.columns and interest_col in X.columns:
            X[f"{prefix}_alignment"] = X[skill_col] * X[interest_col]
            X[f"{prefix}_skill_gap"] = X[skill_col] - X[interest_col]

    for cols, name in [
        (["programming_skill", "web_dev_skill", "mobile_dev_skill"],                "dev_composite"),
        (["data_analytics_skill", "data_science_ml_skill", "data_modeling_skill"],  "data_composite"),
        (["cloud_devops_skill", "cloud_aws", "cloud_azure"],                         "cloud_composite"),
        (["cybersecurity_skill", "siem_experience_score",
          "vuln_assessments_done", "pentest_tools_known_count"],                     "security_composite"),
        (["devops_docker", "devops_kubernetes", "devops_terraform"],                 "devops_composite"),
        (["embedded_c_cpp_skill", "microcontroller_projects_count",
          "rtos_experience_score", "firmware_debugging_skill"],                      "embedded_composite"),
    ]:
        present = [c for c in cols if c in X.columns]
        if present:
            X[name] = X[present].mean(axis=1)

    composite_cols = [c for c in ["dev_composite", "data_composite", "cloud_composite",
                                   "security_composite", "devops_composite",
                                   "embedded_composite"] if c in X.columns]
    if composite_cols:
        X["dominant_domain_score"] = X[composite_cols].max(axis=1)
        X["dominant_domain_idx"]   = X[composite_cols].values.argmax(axis=1).astype(int)

    all_skill_cols = [c for c in X.columns if c.endswith("_skill")]
    if len(all_skill_cols) >= 3:
        X["skill_max"]         = X[all_skill_cols].max(axis=1)
        X["skill_min"]         = X[all_skill_cols].min(axis=1)
        X["skill_range"]       = X["skill_max"] - X["skill_min"]
        X["skill_mean"]        = X[all_skill_cols].mean(axis=1)
        X["skill_std"]         = X[all_skill_cols].std(axis=1)
        X["skills_above_mean"] = X[all_skill_cols].gt(X["skill_mean"], axis=0).sum(axis=1)
        X["skill_focus_ratio"] = X["skill_max"] / (X["skill_mean"] + 1e-6)

    act_cols = [c for c in ["project_count", "github_commits_90d",
                             "internship_experience_count"] if c in X.columns]
    if act_cols:
        X["activity_total"] = X[act_cols].sum(axis=1)
        X["activity_max"]   = X[act_cols].max(axis=1)
        X["is_active"]      = (X["activity_total"] > 0).astype(int)

    if "internship_experience_count" in X.columns:
        X["has_internship"] = (X["internship_experience_count"] > 0).astype(int)

    if "certifications_total" in X.columns:
        X["cert_level_bin"] = pd.cut(
            X["certifications_total"], bins=[-1, 0, 1, 3, float("inf")], labels=[0, 1, 2, 3]
        ).astype(int)
        X["is_certified"] = (X["certifications_total"] > 0).astype(int)

    academic_cols = [c for c in ["math_scores", "cs_fundamentals_scores",
                                  "cognitive_ability_score"] if c in X.columns]
    if academic_cols:
        X["academic_composite"] = X[academic_cols].mean(axis=1)

    if "cgpa" in X.columns:
        X["cgpa_normalized"]  = X["cgpa"] / 10.0
        X["is_top_performer"] = (X["cgpa"] >= 8.5).astype(int)

    for cols, name in [
        (["projects_backend", "projects_frontend", "projects_fullstack"],             "project_dev_total"),
        (["projects_data_analytics", "projects_data_engineering", "projects_ml_ai"], "project_data_total"),
        (["projects_security_defense", "projects_security_offense"],                  "project_security_total"),
        (["projects_cloud", "projects_devops"],                                       "project_cloud_ops"),
        (["projects_mobile_android", "projects_mobile_ios", "projects_mobile_flutter"], "project_mobile_total"),
    ]:
        present = [c for c in cols if c in X.columns]
        if present:
            X[name] = X[present].sum(axis=1)

    for cols, name in [
        (["frontend_react", "frontend_angular"],                       "frontend_stack"),
        (["backend_node", "backend_django", "backend_spring"],         "backend_stack"),
        (["data_tool_spark", "data_tool_airflow", "data_tool_kafka"],  "data_stack"),
        (["security_tool_siem", "security_tool_wireshark", "security_tool_burpsuite"], "security_stack"),
        (["mobile_kotlin", "mobile_flutter"],                          "mobile_stack"),
        (["observability_prometheus", "observability_grafana"],        "observability_stack"),
    ]:
        present = [c for c in cols if c in X.columns]
        if present:
            X[name] = X[present].sum(axis=1)

    all_sec = [c for c in ["cybersecurity_skill", "siem_experience_score",
                            "vuln_assessments_done", "pentest_tools_known_count",
                            "incident_response_cases", "compliance_frameworks_known_count"] if c in X.columns]
    if all_sec:
        X["security_total"]      = X[all_sec].sum(axis=1)
        X["is_security_focused"] = (X["security_total"] >= 5).astype(int)

    pentest_cols = [c for c in ["pentest_tools_known_count", "security_tool_burpsuite",
                                 "projects_security_offense"] if c in X.columns]
    if pentest_cols:
        X["pentest_focused"] = X[pentest_cols].sum(axis=1)

    defense_cols = [c for c in ["siem_experience_score", "incident_response_cases",
                                 "security_tool_siem", "projects_security_defense"] if c in X.columns]
    if defense_cols:
        X["defensive_focused"] = X[defense_cols].sum(axis=1)

    soft_cols = [c for c in ["teamwork_behavior", "communication_skill",
                              "learning_motivation", "professional_discipline_score"] if c in X.columns]
    if soft_cols:
        X["soft_skills_total"] = X[soft_cols].sum(axis=1)
        X["soft_skills_mean"]  = X[soft_cols].mean(axis=1)
        X["is_strong_soft"]    = (X["soft_skills_mean"] >= 7).astype(int)

    return X

X_train_fe = add_advanced_engineered_features(X_train)
X_val_fe   = add_advanced_engineered_features(X_val)
X_test_fe  = add_advanced_engineered_features(X_test)
print(f"  After FE: {X_train_fe.shape[1]} features")

# Correlation-based feature selection
def find_high_corr_drops(X_train_df, threshold=0.85):
    numeric_cols = [c for c in X_train_df.columns if pd.api.types.is_numeric_dtype(X_train_df[c])]
    X_num = X_train_df[numeric_cols].copy()
    if X_num.shape[1] < 2:
        return []
    corr  = X_num.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        partners = upper.index[upper[col] > threshold].tolist()
        for partner in partners:
            if X_num[col].std() < X_num[partner].std():
                to_drop.add(col)
            else:
                to_drop.add(partner)
    return list(to_drop)

drop_corr_cols = find_high_corr_drops(X_train_fe, threshold=0.85)
print(f"  Dropping {len(drop_corr_cols)} correlated features")

X_train_sel = X_train_fe.drop(columns=drop_corr_cols, errors="ignore")
X_val_sel   = X_val_fe.drop(columns=drop_corr_cols, errors="ignore")
X_test_sel  = X_test_fe.drop(columns=drop_corr_cols, errors="ignore")

(OUT_DIR / "dropped_correlated_features.json").write_text(
    json.dumps({"threshold": 0.85, "dropped_features": drop_corr_cols}, indent=2)
)
(OUT_DIR / "selected_features.json").write_text(
    json.dumps({"selected_features": X_train_sel.columns.tolist()}, indent=2)
)

# ── LAYER 4 — Preprocessing ───────────────────────────────────────────────────
print("\n[LAYER 4] Preprocessing...")
numeric_cols    = [c for c in X_train_sel.columns if pd.api.types.is_numeric_dtype(X_train_sel[c])]
categorical_cols = [c for c in X_train_sel.columns if c not in numeric_cols]

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_pipe, numeric_cols),
        ("cat", categorical_pipe, categorical_cols),
    ],
    remainder="drop",
)

X_train_proc = preprocess.fit_transform(X_train_sel)
X_val_proc   = preprocess.transform(X_val_sel)
X_test_proc  = preprocess.transform(X_test_sel)
print(f"  Processed shape: {X_train_proc.shape}")

joblib.dump(preprocess, OUT_DIR / "preprocess.joblib")
final_feature_list = X_train_sel.columns.tolist()
(OUT_DIR / "final_feature_list.json").write_text(
    json.dumps({"final_feature_list": final_feature_list}, indent=2)
)
(OUT_DIR / "feature_order.json").write_text(json.dumps(final_feature_list, indent=2))

# ── Role Merge Map + Label Encoding ──────────────────────────────────────────
ROLE_MERGE_MAP = {
    "Cloud Engineer":            "Cloud Engineer & Platform/SRE",
    "Platform Engineer":         "Cloud Engineer & Platform/SRE",
    "Site Reliability Engineer": "Cloud Engineer & Platform/SRE",
    "AI Engineer":               "Data Scientist / ML / AI Engineer",
    "Data Scientist":            "Data Scientist / ML / AI Engineer",
    "Machine Learning Engineer": "Data Scientist / ML / AI Engineer",
    "BI Analyst":                "Data Scientist / ML / AI Engineer",
    "Analytics Engineer":        "Data & Analytics Engineer",
    "Data Engineer":             "Data & Analytics Engineer",
    "SOC Analyst":               "Security Operations Analyst",
    "Security Analyst":          "Security Operations Analyst",
    "Security Operations Engineer": "Security Operations Analyst",
    "Cybersecurity Engineer":    "Cybersecurity & Cloud Security Engineer",
    "Cloud Security Engineer":   "Cybersecurity & Cloud Security Engineer",
    "QA Engineer":               "QA & Automation Test Engineer",
    "Automation Test Engineer":  "QA & Automation Test Engineer",
}

(OUT_DIR / "role_merge_map.json").write_text(json.dumps(ROLE_MERGE_MAP, indent=2))

y_train_mapped = y_train.replace(ROLE_MERGE_MAP)
y_val_mapped   = y_val.replace(ROLE_MERGE_MAP)
y_test_mapped  = y_test.replace(ROLE_MERGE_MAP)

label_encoder = LabelEncoder()
y_train_enc   = label_encoder.fit_transform(y_train_mapped.astype(str))
y_val_enc     = label_encoder.transform(y_val_mapped.astype(str))
y_test_enc    = label_encoder.transform(y_test_mapped.astype(str))

n_classes = len(label_encoder.classes_)
print(f"\n  Final classes: {n_classes}")
for i, r in enumerate(label_encoder.classes_):
    print(f"    {i:>2}. {r}")

joblib.dump(label_encoder, OUT_DIR / "label_encoder.joblib")

# Rebuild role_to_domain with merged roles
with open(OUT_DIR / "role_to_domain.json") as f:
    original_r2d = json.load(f)
role_to_domain_merged = {}
for old_role, domain in original_r2d.items():
    new_role = ROLE_MERGE_MAP.get(old_role, old_role)
    role_to_domain_merged[new_role] = domain
(OUT_DIR / "role_to_domain.json").write_text(json.dumps(role_to_domain_merged, indent=2))

# ── LAYER 5 — Model Training ──────────────────────────────────────────────────
print("\n[LAYER 5] Training models...")

def aligned_proba(model, X, n_classes):
    proba = model.predict_proba(X)
    if hasattr(model, "classes_"):
        aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
        aligned[:, model.classes_] = proba
        return aligned
    return proba

def eval_on_val_metrics(model, X_tr, y_tr, X_v, y_v, k=3):
    model.fit(X_tr, y_tr)
    pred  = model.predict(X_v)
    nc    = len(label_encoder.classes_)
    metrics = {
        "accuracy":        float(accuracy_score(y_v, pred)),
        "macro_f1":        float(f1_score(y_v, pred, average="macro")),
        "macro_precision": float(precision_score(y_v, pred, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(y_v, pred, average="macro", zero_division=0)),
        "top3_acc": None, "top3_f1": None, "log_loss": None,
    }
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_v)
        metrics["log_loss"] = float(log_loss(y_v, proba, labels=np.arange(nc)))
        metrics["top3_acc"] = float(top_k_accuracy_score(y_v, proba, k=k, labels=np.arange(nc)))
        topk_idx = np.argsort(proba, axis=1)[:, -k:]
        hit = np.array([y_v[i] in topk_idx[i] for i in range(len(y_v))], dtype=int)
        metrics["top3_f1"] = float(f1_score(np.ones_like(hit), hit, average="binary"))
    return metrics, model

# LightGBM
print("  Training LightGBM...")
t0 = time.time()
lgbm_model = lgb.LGBMClassifier(
    n_estimators=800, learning_rate=0.07, num_leaves=80, max_depth=-1,
    min_child_samples=25, subsample=0.85, colsample_bytree=0.85,
    reg_alpha=0.1, reg_lambda=1.0, class_weight="balanced",
    random_state=SEED, verbosity=-1, n_jobs=-1,
)
lgbm_metrics, lgbm_fitted = eval_on_val_metrics(lgbm_model, X_train_proc, y_train_enc, X_val_proc, y_val_enc)
print(f"    Val accuracy: {lgbm_metrics['accuracy']:.4f}  F1: {lgbm_metrics['macro_f1']:.4f}  [{time.time()-t0:.1f}s]")

# XGBoost
print("  Training XGBoost...")
t0 = time.time()
xgb_model = XGBClassifier(
    n_estimators=800, learning_rate=0.07, max_depth=5, min_child_weight=4,
    subsample=0.75, colsample_bytree=0.75, colsample_bylevel=0.6,
    gamma=0.2, reg_alpha=0.2, reg_lambda=3.0, eval_metric="mlogloss",
    use_label_encoder=False, random_state=SEED, n_jobs=-1, verbosity=0,
)
xgb_metrics, xgb_fitted = eval_on_val_metrics(xgb_model, X_train_proc, y_train_enc, X_val_proc, y_val_enc)
print(f"    Val accuracy: {xgb_metrics['accuracy']:.4f}  F1: {xgb_metrics['macro_f1']:.4f}  [{time.time()-t0:.1f}s]")

# CatBoost
print("  Training CatBoost...")
t0 = time.time()
catboost_model = CatBoostClassifier(
    iterations=600, learning_rate=0.07, depth=6, l2_leaf_reg=4.0,
    border_count=64, random_strength=1.0, bagging_temperature=0.8,
    auto_class_weights="Balanced", od_type="Iter", od_wait=40,
    task_type="CPU", random_seed=SEED, verbose=0,
)
catboost_metrics, catboost_fitted = eval_on_val_metrics(catboost_model, X_train_proc, y_train_enc, X_val_proc, y_val_enc)
print(f"    Val accuracy: {catboost_metrics['accuracy']:.4f}  F1: {catboost_metrics['macro_f1']:.4f}  [{time.time()-t0:.1f}s]")

# Logistic Regression
print("  Training Logistic Regression...")
t0 = time.time()
lr_model = LogisticRegression(C=0.8, penalty="l2", solver="lbfgs", max_iter=1000,
                               class_weight="balanced", random_state=SEED)
lr_metrics, lr_fitted = eval_on_val_metrics(lr_model, X_train_proc, y_train_enc, X_val_proc, y_val_enc)
print(f"    Val accuracy: {lr_metrics['accuracy']:.4f}  F1: {lr_metrics['macro_f1']:.4f}  [{time.time()-t0:.1f}s]")

# ── OOF Stacking ──────────────────────────────────────────────────────────────
print("\n  Building OOF Stacking Ensemble...")

def build_oof_stack_features(base_models, X_tr, y_tr, X_v, n_splits=5, seed=42):
    nc     = len(label_encoder.classes_)
    kf     = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    n_base = len(base_models)
    oof_tr = np.zeros((X_tr.shape[0], n_base * nc))
    oof_v  = np.zeros((X_v.shape[0], n_base * nc))
    for fold, (ti, vi) in enumerate(kf.split(X_tr, y_tr)):
        print(f"    Fold {fold+1}/{n_splits}...")
        for j, (name, mdl) in enumerate(base_models.items()):
            m = clone(mdl)
            m.fit(X_tr[ti], y_tr[ti])
            oof_tr[vi, j*nc:(j+1)*nc] = aligned_proba(m, X_tr[vi], nc)
            oof_v[:, j*nc:(j+1)*nc]  += aligned_proba(m, X_v, nc) / n_splits
    return oof_tr, oof_v

base_models = {"xgb": xgb_fitted, "lgbm": lgbm_fitted, "cat": catboost_fitted, "lr": lr_fitted}

t0 = time.time()
meta_train, meta_val = build_oof_stack_features(
    base_models, X_train_proc, y_train_enc, X_val_proc, n_splits=5, seed=SEED
)

meta_model = LogisticRegression(C=1.0, solver="lbfgs", penalty="l2",
                                 class_weight="balanced", max_iter=1000, random_state=SEED)
meta_model.fit(meta_train, y_train_enc)
stack_elapsed = time.time() - t0

val_pred  = meta_model.predict(meta_val)
val_proba = meta_model.predict_proba(meta_val)

def top3_f1_from_proba(y_true, proba, k=3):
    topk_idx = np.argsort(proba, axis=1)[:, -k:]
    hit = np.array([y_true[i] in topk_idx[i] for i in range(len(y_true))], dtype=int)
    return float(f1_score(np.ones_like(hit), hit, average="binary"))

stack_metrics = {
    "accuracy":        float(accuracy_score(y_val_enc, val_pred)),
    "macro_f1":        float(f1_score(y_val_enc, val_pred, average="macro")),
    "macro_precision": float(precision_score(y_val_enc, val_pred, average="macro", zero_division=0)),
    "macro_recall":    float(recall_score(y_val_enc, val_pred, average="macro", zero_division=0)),
    "top3_acc":        float(top_k_accuracy_score(y_val_enc, val_proba, k=3, labels=np.arange(n_classes))),
    "top3_f1":         float(top3_f1_from_proba(y_val_enc, val_proba, k=3)),
    "log_loss":        float(log_loss(y_val_enc, val_proba, labels=np.arange(n_classes))),
}

print(f"\n  STACKING ENSEMBLE VAL RESULTS:")
for k, v in stack_metrics.items():
    print(f"    {k}: {v:.6f}")
print(f"  Training time: {stack_elapsed:.1f}s")

bundle = {"base_models": base_models, "meta_model": meta_model}
joblib.dump(bundle, OUT_DIR / "stacking_ensemble.joblib")
(OUT_DIR / "stacking_val_results.json").write_text(json.dumps(stack_metrics, indent=2))

# ── LAYER 6 — Test Evaluation ─────────────────────────────────────────────────
print("\n[LAYER 6] Evaluating on test set...")

def expected_calibration_error(proba, y_true, n_bins=15):
    conf = proba.max(axis=1)
    pred = proba.argmax(axis=1)
    hit  = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for i in range(n_bins):
        mask = (conf >= bins[i]) & (conf < bins[i+1])
        if mask.sum() > 0:
            ece += mask.sum() / len(conf) * abs(hit[mask].mean() - conf[mask].mean())
    return float(ece)

def predict_proba_ensemble(bundle, X_te):
    nc    = len(label_encoder.classes_)
    parts = []
    for name, base in bundle["base_models"].items():
        parts.append(aligned_proba(base, X_te, nc))
    meta_X = np.hstack(parts)
    proba  = bundle["meta_model"].predict_proba(meta_X)
    pred   = bundle["meta_model"].predict(meta_X)
    return proba, pred

ens_proba_test, ens_pred_test = predict_proba_ensemble(bundle, X_test_proc)

test_metrics = {
    "accuracy":        float(accuracy_score(y_test_enc, ens_pred_test)),
    "macro_f1":        float(f1_score(y_test_enc, ens_pred_test, average="macro")),
    "macro_precision": float(precision_score(y_test_enc, ens_pred_test, average="macro", zero_division=0)),
    "macro_recall":    float(recall_score(y_test_enc, ens_pred_test, average="macro", zero_division=0)),
    "top3_acc":        float(top_k_accuracy_score(y_test_enc, ens_proba_test, k=3, labels=np.arange(n_classes))),
    "top3_f1":         float(top3_f1_from_proba(y_test_enc, ens_proba_test, k=3)),
    "log_loss":        float(log_loss(y_test_enc, ens_proba_test, labels=np.arange(n_classes))),
    "ece":             expected_calibration_error(ens_proba_test, y_test_enc),
    "mean_confidence": float(ens_proba_test.max(axis=1).mean()),
    "pct_above_99":    float((ens_proba_test.max(axis=1) > 0.99).mean() * 100),
}

print("\n  TEST SET RESULTS:")
for k, v in test_metrics.items():
    print(f"    {k}: {v:.6f}")

(OUT_DIR / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))

report_txt = classification_report(y_test_enc, ens_pred_test,
                                    target_names=label_encoder.classes_, zero_division=0)
(OUT_DIR / "ensemble_classification_report_test.txt").write_text(report_txt)

# Save model metadata
model_meta = {
    "version":       "1.0.0",
    "trained_at":    datetime.datetime.now().isoformat(),
    "n_classes":     int(n_classes),
    "base_models":   list(base_models.keys()),
    "meta_model":    "LogisticRegression",
    "stacking_type": "OOF-5fold",
    "val_macro_f1":  round(stack_metrics["macro_f1"], 6),
    "val_accuracy":  round(stack_metrics["accuracy"], 6),
    "test_macro_f1": round(test_metrics["macro_f1"], 6),
    "test_accuracy": round(test_metrics["accuracy"], 6),
}
(OUT_DIR / "model_meta.json").write_text(json.dumps(model_meta, indent=2))

print("\n" + "=" * 60)
print("  TRAINING COMPLETE!")
print(f"  Test Accuracy : {test_metrics['accuracy']:.4f}")
print(f"  Test Macro F1 : {test_metrics['macro_f1']:.4f}")
print(f"  Test Top-3 Acc: {test_metrics['top3_acc']:.4f}")
print("  All artifacts saved to artifacts_layer/")
print("=" * 60)
