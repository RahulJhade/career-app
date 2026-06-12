"""
Generate SHAP explainer from the trained LightGBM base model.
Run once: python generate_shap.py
"""
import json
import joblib
import numpy as np
import shap
from pathlib import Path

OUT_DIR = Path("artifacts_layer")

print("Loading model artifacts...")
ensemble = joblib.load(OUT_DIR / "stacking_ensemble.joblib")
label_encoder = joblib.load(OUT_DIR / "label_encoder.joblib")

# Use LightGBM as the SHAP base — it's the strongest tree model
# and TreeExplainer is exact + fast for gradient-boosted trees
lgbm_model = ensemble["base_models"]["lgbm"]

print(f"Creating TreeExplainer for LightGBM ({len(label_encoder.classes_)} classes)...")
explainer = shap.TreeExplainer(lgbm_model)

print("Saving to artifacts_layer/shap_explainer.joblib...")
joblib.dump(explainer, OUT_DIR / "shap_explainer.joblib")

# Quick verification
feature_order = json.loads((OUT_DIR / "feature_order.json").read_text())
print(f"\n✓ SHAP explainer saved successfully!")
print(f"  Model: LightGBM ({lgbm_model.n_estimators} estimators)")
print(f"  Classes: {len(label_encoder.classes_)}")
print(f"  Features: {len(feature_order)}")
