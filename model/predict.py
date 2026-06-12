"""
model/predict.py
─────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS:
  Single entry point for all prediction logic.  Keeps app.py clean.
  All prediction-related logic (probability extraction, top-k decoding,
  confidence thresholds, domain mapping) lives here.

WHY PROBABILITY-BASED PREDICTIONS:
  Hard argmax gives only the top-1 class with zero calibration info.
  Returning the full probability vector lets us:
    1. Show top-3 alternatives with real confidence percentages.
    2. Flag low-confidence predictions (< 30%) with a warning.
    3. Support what-if simulation (change one feature, compare distributions).
    4. Feed downstream analytics (confidence over time trends).
─────────────────────────────────────────────────────────────────────────────
"""

import logging
import numpy as np

logger = logging.getLogger(__name__)

# Domain → display colour and icon (used by frontend)
DOMAIN_COLORS = {
    "Software Engineering":                   "#3b82f6",
    "Data & Artificial Intelligence":         "#8b5cf6",
    "Cybersecurity":                          "#ef4444",
    "Cloud, DevOps & Platform Engineering":  "#06b6d4",
    "UI/UX & Product":                        "#f59e0b",
    "Quality Assurance & Testing":            "#10b981",
    "Systems & Infrastructure":               "#6366f1",
}
DOMAIN_ICONS = {
    "Software Engineering":                   "💻",
    "Data & Artificial Intelligence":         "🤖",
    "Cybersecurity":                          "🔐",
    "Cloud, DevOps & Platform Engineering":  "☁️",
    "UI/UX & Product":                        "🎨",
    "Quality Assurance & Testing":            "✅",
    "Systems & Infrastructure":               "🖥️",
}

# Minimum confidence below which we emit a low-confidence warning to the user
LOW_CONFIDENCE_THRESHOLD = 30.0


def _aligned_proba(model, X: np.ndarray, n_classes: int) -> np.ndarray:
    """
    WHY: Some sklearn-wrapped models (e.g. fitted on a subset of classes during
    OOF folds) may have model.classes_ = [0,2,5,...] not [0,1,2,...].
    We must expand to a full (n_samples, n_classes) array or the meta-model
    stacking will receive wrong-shaped / wrong-column inputs.
    """
    proba = model.predict_proba(X)
    if hasattr(model, "classes_"):
        aligned = np.zeros((proba.shape[0], n_classes), dtype=float)
        aligned[:, model.classes_] = proba
        return aligned
    return proba


def predict_top3(
    raw_input: dict,
    *,
    ensemble: dict,
    preprocessor,
    label_encoder,
    feature_order: list,
    drop_corr: list,
    role_to_domain: dict,
) -> list[dict]:
    """
    Full inference pipeline:
      raw_dict → preprocess → base-model probas → meta-model → top-3 decode

    Returns:
      List of 3 dicts, each with keys:
        role, domain, probability (%), color, icon

    FAILURE PREVENTION:
      - Zero-vector input   → model still returns a valid probability distribution
      - Missing features    → align_and_preprocess fills with 0
      - Model single-class  → only happens if trained wrong; logs a warning
    """
    from model.preprocess import align_and_preprocess  # local import avoids circular

    n_classes = len(label_encoder.classes_)

    # Step 1 — Preprocess
    X = align_and_preprocess(
        raw_input, feature_order, drop_corr, preprocessor, log_shapes=True
    )

    # Step 2 — Base model probabilities (stacking layer)
    base_parts = [
        _aligned_proba(m, X, n_classes)
        for m in ensemble["base_models"].values()
    ]
    meta_X = np.hstack(base_parts)           # (1, n_base * n_classes)

    # Step 3 — Meta-model final probabilities
    proba = ensemble["meta_model"].predict_proba(meta_X)[0]   # (n_classes,)

    # Sanity checks
    if not np.isfinite(proba).all():
        logger.error("Non-finite probabilities detected — replacing with uniform")
        proba = np.ones(n_classes) / n_classes

    logger.debug(
        "predict_top3 | top prob=%.3f  2nd=%.3f  3rd=%.3f",
        *sorted(proba, reverse=True)[:3],
    )

    # Step 4 — Decode top-3
    top3_idx  = np.argsort(proba)[::-1][:3]
    results   = []
    for rank, idx in enumerate(top3_idx):
        role   = label_encoder.classes_[idx]
        domain = role_to_domain.get(role, "Unknown")
        results.append({
            "rank":        rank + 1,
            "role":        role,
            "domain":      domain,
            "probability": round(float(proba[idx]) * 100, 1),
            "color":       DOMAIN_COLORS.get(domain, "#6b7280"),
            "icon":        DOMAIN_ICONS.get(domain, "💼"),
        })

    # Low-confidence warning flag
    top_conf = results[0]["probability"]
    if top_conf < LOW_CONFIDENCE_THRESHOLD:
        logger.warning(
            "Low-confidence prediction: %.1f%% for '%s'. "
            "User should add more skill signals.",
            top_conf, results[0]["role"],
        )
        for r in results:
            r["low_confidence"] = True
    else:
        for r in results:
            r["low_confidence"] = False

    return results


def predict_whatif(
    base_input: dict,
    changed_field: str,
    new_value: float,
    **kwargs,
) -> list[dict]:
    """
    What-if simulator: change exactly one field, return new top-3.
    The caller passes the same kwargs as predict_top3.
    WHY: Lets users see 'if I raise my programming_skill to 9, does DS/ML appear?'
    """
    modified = {**base_input, changed_field: new_value}
    return predict_top3(modified, **kwargs)
