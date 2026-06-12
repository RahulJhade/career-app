"""
model/preprocess.py
─────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS:
  Training used a ColumnTransformer (StandardScaler on numerics, OHE on cats).
  Inference MUST apply the IDENTICAL transformation — same fitted scaler,
  same fitted encoder, same column order — or predictions are garbage.

  This module is the single source of truth for:
    1. Feature engineering  (add derived columns)
    2. Feature alignment    (reindex to exactly the trained feature_order)
    3. Preprocessing        (apply the saved sklearn pipeline)

  Separation from predict.py exists so tests, notebooks, and batch jobs
  can call the preprocessor independently without touching the model.
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─── Feature Engineering ─────────────────────────────────────────────────────
# WHY: The model was trained on raw features PLUS derived composite scores,
#      alignment scores, domain flags, etc.  If we don't reproduce these at
#      inference time the feature_order will be full of zeros for those columns
#      and the model will produce degraded / wrong predictions.

def add_features(X_in: pd.DataFrame) -> pd.DataFrame:
    """Reproduce every engineered feature from training — ORDER MATTERS."""
    X = X_in.copy()

    # ── Skill × Interest Alignment scores ────────────────────────────────────
    # WHY: A high ML skill + high ML interest = strong signal for DS/ML roles.
    #      Product captures synergy; difference captures "knows but dislikes".
    for skill_col, interest_col, prefix in [
        ("programming_skill",       "interest_dev_overall",             "prog"),
        ("data_analytics_skill",    "interest_data_overall",            "data_analytics"),
        ("data_science_ml_skill",   "interest_data_overall",            "data_science"),
        ("cloud_devops_skill",      "interest_cloud_infra_overall",     "cloud"),
        ("cybersecurity_skill",     "interest_cybersecurity",           "cyber"),
        ("ui_ux_design_skill",      "interest_ui_ux_design",            "uiux"),
        ("business_analysis_skill", "interest_business_and_management", "biz"),
    ]:
        if skill_col in X and interest_col in X:
            X[f"{prefix}_alignment"] = X[skill_col] * X[interest_col]
            X[f"{prefix}_skill_gap"] = X[skill_col] - X[interest_col]

    # ── Domain Composite scores ───────────────────────────────────────────────
    # WHY: Individual skill scores are noisy. A composite mean gives the model
    #      a robust single signal per domain.
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
        present = [c for c in cols if c in X]
        X[name] = X[present].mean(axis=1) if present else 0

    # ── Dominant domain ───────────────────────────────────────────────────────
    # WHY: Tells the model which domain is the user's strongest — acts as a
    #      global context signal that prevents wishy-washy predictions.
    composite_cols = [c for c in ["dev_composite", "data_composite", "cloud_composite",
                                   "security_composite", "devops_composite",
                                   "embedded_composite"] if c in X]
    if composite_cols:
        X["dominant_domain_score"] = X[composite_cols].max(axis=1)
        X["dominant_domain_idx"]   = X[composite_cols].values.argmax(axis=1).astype(int)

    # ── Skill profile statistics ──────────────────────────────────────────────
    # WHY: skill_focus_ratio separates specialists (high ratio) from generalists.
    #      This strongly discriminates between senior engineers vs. full-stack juniors.
    skill_cols = [c for c in X if c.endswith("_skill")]
    if len(skill_cols) >= 3:
        X["skill_max"]         = X[skill_cols].max(axis=1)
        X["skill_min"]         = X[skill_cols].min(axis=1)
        X["skill_range"]       = X["skill_max"] - X["skill_min"]
        X["skill_mean"]        = X[skill_cols].mean(axis=1)
        X["skill_std"]         = X[skill_cols].std(axis=1)
        X["skills_above_mean"] = X[skill_cols].gt(X["skill_mean"], axis=0).sum(axis=1)
        X["skill_focus_ratio"] = X["skill_max"] / (X["skill_mean"] + 1e-6)

    # ── Activity signals ──────────────────────────────────────────────────────
    act_cols = [c for c in ["project_count", "github_commits_90d",
                              "internship_experience_count"] if c in X]
    if act_cols:
        X["activity_total"] = X[act_cols].sum(axis=1)
        X["activity_max"]   = X[act_cols].max(axis=1)
        X["is_active"]      = (X["activity_total"] > 0).astype(int)

    if "internship_experience_count" in X:
        X["has_internship"] = (X["internship_experience_count"] > 0).astype(int)

    # ── Certification tier ────────────────────────────────────────────────────
    if "certifications_total" in X:
        X["cert_level_bin"] = pd.cut(
            X["certifications_total"], bins=[-1, 0, 1, 3, float("inf")], labels=[0, 1, 2, 3]
        ).astype(int)
        X["is_certified"] = (X["certifications_total"] > 0).astype(int)

    # ── Academic composite ────────────────────────────────────────────────────
    acad_cols = [c for c in ["math_scores", "cs_fundamentals_scores",
                               "cognitive_ability_score"] if c in X]
    if acad_cols:
        X["academic_composite"] = X[acad_cols].mean(axis=1)

    if "cgpa" in X:
        X["cgpa_normalized"]  = X["cgpa"] / 10.0
        X["is_top_performer"] = (X["cgpa"] >= 8.5).astype(int)

    # ── Project category totals ───────────────────────────────────────────────
    for cols, name in [
        (["projects_backend", "projects_frontend", "projects_fullstack"],             "project_dev_total"),
        (["projects_data_analytics", "projects_data_engineering", "projects_ml_ai"], "project_data_total"),
        (["projects_security_defense", "projects_security_offense"],                  "project_security_total"),
        (["projects_cloud", "projects_devops"],                                       "project_cloud_ops"),
        (["projects_mobile_android", "projects_mobile_ios", "projects_mobile_flutter"], "project_mobile_total"),
    ]:
        present = [c for c in cols if c in X]
        X[name] = X[present].sum(axis=1) if present else 0

    # ── Tool stack counts ─────────────────────────────────────────────────────
    for cols, name in [
        (["frontend_react", "frontend_angular"],                       "frontend_stack"),
        (["backend_node", "backend_django", "backend_spring"],         "backend_stack"),
        (["data_tool_spark", "data_tool_airflow", "data_tool_kafka"],  "data_stack"),
        (["security_tool_siem", "security_tool_wireshark",
          "security_tool_burpsuite"],                                  "security_stack"),
        (["mobile_kotlin", "mobile_flutter"],                          "mobile_stack"),
        (["observability_prometheus", "observability_grafana"],        "observability_stack"),
    ]:
        present = [c for c in cols if c in X]
        X[name] = X[present].sum(axis=1) if present else 0

    # ── Security specialisation signals ───────────────────────────────────────
    sec_cols = [c for c in ["cybersecurity_skill", "siem_experience_score",
                              "vuln_assessments_done", "pentest_tools_known_count",
                              "incident_response_cases",
                              "compliance_frameworks_known_count"] if c in X]
    if sec_cols:
        X["security_total"]      = X[sec_cols].sum(axis=1)
        X["is_security_focused"] = (X["security_total"] >= 5).astype(int)

    pentest_cols = [c for c in ["pentest_tools_known_count", "security_tool_burpsuite",
                                  "projects_security_offense"] if c in X]
    if pentest_cols:
        X["pentest_focused"] = X[pentest_cols].sum(axis=1)

    defense_cols = [c for c in ["siem_experience_score", "incident_response_cases",
                                  "security_tool_siem", "projects_security_defense"] if c in X]
    if defense_cols:
        X["defensive_focused"] = X[defense_cols].sum(axis=1)

    # ── Soft skills composite ─────────────────────────────────────────────────
    soft_cols = [c for c in ["teamwork_behavior", "communication_skill",
                               "learning_motivation", "professional_discipline_score"] if c in X]
    if soft_cols:
        X["soft_skills_total"] = X[soft_cols].sum(axis=1)
        X["soft_skills_mean"]  = X[soft_cols].mean(axis=1)
        X["is_strong_soft"]    = (X["soft_skills_mean"] >= 7).astype(int)

    return X


def align_and_preprocess(
    raw_input: dict,
    feature_order: list,
    drop_corr: list,
    preprocessor,
    log_shapes: bool = True,
) -> np.ndarray:
    """
    Full inference pipeline:
      raw dict → DataFrame → feature engineering → drop correlated →
      reindex to feature_order (missing cols filled with 0) → preprocessor.transform

    WHY feature alignment exists:
      sklearn's ColumnTransformer was fitted on a fixed column order.
      If columns arrive in a different order, or new columns appear,
      or training columns are missing, the transform produces wrong results.
      fill_value=0 is safe because: missing features = "user has none of this".

    FAILURE PREVENTION:
      - Extra keys in raw_input → safely ignored (not in feature_order)
      - Missing keys          → filled with 0 (explicit, not NaN-silent)
      - All-zero input        → handled gracefully (model returns soft proba)
    """
    df = pd.DataFrame([raw_input])
    df_fe = add_features(df)

    # Drop correlated features (same ones dropped during training)
    df_sel = df_fe.drop(columns=drop_corr, errors="ignore")

    # Align to exact training feature order — this is the critical step
    df_aligned = df_sel.reindex(columns=feature_order, fill_value=0)

    if log_shapes:
        logger.debug(
            "align_and_preprocess | raw_keys=%d fe_cols=%d aligned_cols=%d",
            len(raw_input), len(df_fe.columns), len(df_aligned.columns),
        )

    X = preprocessor.transform(df_aligned)

    if log_shapes:
        logger.debug("preprocessed shape: %s", X.shape)

    return X
