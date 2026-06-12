"""
app.py — CareerAI Flask Application (v2 — Clean Architecture)
─────────────────────────────────────────────────────────────────────────────
Architecture:
  Request → validate (utils/validators.py)
           → predict  (model/predict.py)
                ├─ preprocess (model/preprocess.py)
                └─ ensemble   (artifacts_layer/*.joblib)
           → enrich   (skill_gap, peer_compare, readiness, roadmap, salary…)
           → Response JSON

This file is intentionally thin.  Business logic lives in model/ and utils/.
─────────────────────────────────────────────────────────────────────────────
"""

import io
import json
import logging
import os
import re
import sqlite3
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

# Load .env (GEMINI_API_KEY)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("career_ai")

# ── Optional deps ─────────────────────────────────────────────────────────────
try:
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logger.warning("pdfplumber not installed — PDF upload disabled")

try:
    import shap as shap_lib
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (HRFlowable, Paragraph, SimpleDocTemplate,
                                     Spacer, Table, TableStyle)
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("reportlab not installed — PDF export disabled")

# ── App init ──────────────────────────────────────────────────────────────────
app     = Flask(__name__)
OUT_DIR = Path("artifacts_layer")

# ── Load artifacts (done ONCE at startup) ────────────────────────────────────
logger.info("Loading model artifacts…")
ensemble      = joblib.load(OUT_DIR / "stacking_ensemble.joblib")
preprocessor  = joblib.load(OUT_DIR / "preprocess.joblib")
label_encoder = joblib.load(OUT_DIR / "label_encoder.joblib")

feature_order  = json.loads((OUT_DIR / "feature_order.json").read_text())
drop_corr      = json.loads((OUT_DIR / "dropped_correlated_features.json").read_text())["dropped_features"]
role_to_domain = json.loads((OUT_DIR / "role_to_domain.json").read_text())
merge_map      = json.loads((OUT_DIR / "role_merge_map.json").read_text())
model_meta     = json.loads((OUT_DIR / "model_meta.json").read_text())
test_metrics   = json.loads((OUT_DIR / "test_metrics.json").read_text())
analytics_data = json.loads((OUT_DIR / "analytics_data.json").read_text())
feature_imp    = json.loads((OUT_DIR / "feature_importance.json").read_text())
role_profiles  = json.loads((OUT_DIR / "role_skill_profiles.json").read_text())
role_pctiles   = json.loads((OUT_DIR / "role_percentiles.json").read_text())
roadmaps       = json.loads((OUT_DIR / "roadmaps.json").read_text())
salary_data    = json.loads((OUT_DIR / "salary_data.json").read_text())
radar_profiles = json.loads((OUT_DIR / "radar_profiles.json").read_text())
companies_data = json.loads((OUT_DIR / "companies.json").read_text())
action_res     = json.loads((OUT_DIR / "action_plan_resources.json").read_text())
logger.info("Artifacts loaded — %d classes, %d features",
            len(label_encoder.classes_), len(feature_order))

# ── SHAP explainer — loaded in background to avoid startup blocking ──────────
# The file is ~1.16 GB; loading it synchronously would stall the first request
# and crash the app if memory is tight.  We start a daemon thread immediately
# so the app is ready to serve requests in <1 s while SHAP warms up.
shap_explainer = None
_shap_semaphore = threading.Semaphore(1)   # only 1 live SHAP computation at a time
_shap_loaded = threading.Event()           # set once loading completes (or fails)

def _load_shap_background():
    global shap_explainer
    path = OUT_DIR / "shap_explainer.joblib"
    if not (SHAP_AVAILABLE and path.exists()):
        _shap_loaded.set()
        return
    try:
        size_mb = path.stat().st_size / 1_000_000
        logger.info("SHAP background load started (%.0f MB) …", size_mb)
        shap_explainer = joblib.load(path)
        logger.info("SHAP explainer ready ✓")
    except Exception as exc:
        logger.warning("SHAP background load failed: %s", exc)
    finally:
        _shap_loaded.set()

threading.Thread(target=_load_shap_background, daemon=True,
                 name="shap-loader").start()

# ── Groq LLM (free, fast — replaces Gemini) ──────────────────────────────────
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
groq_client = None
GROQ_MODEL = "llama-3.3-70b-versatile"  # Free tier, very capable

if GROQ_API_KEY:
    try:
        from groq import Groq
        groq_client = Groq(api_key=GROQ_API_KEY)
        # Quick connectivity test
        _test = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "Say OK"}],
            max_tokens=5,
        )
        logger.info("Groq LLM (%s) loaded ✓", GROQ_MODEL)
    except Exception as exc:
        logger.warning("Groq init failed: %s", exc)
        groq_client = None
else:
    logger.warning("GROQ_API_KEY not set — AI Advisor will use fallback mode")

# ── LLM Prediction Cache (avoid repeat API calls for same profile) ────────────
_llm_pred_cache: dict = {}   # {profile_hash: (timestamp, result_dict)}
LLM_PRED_CACHE_TTL = 3600   # 1 hour

# ── Live Jobs Cache ────────────────────────────────────────────────────────
_jobs_cache: dict = {}   # {role: (timestamp, jobs_list)}
JOBS_CACHE_TTL = 3600   # 1 hour

ROLE_TO_SEARCH = {
    "Software Engineer":          "software engineer developer",
    "Data Scientist":             "data scientist",
    "Data Analyst":               "data analyst",
    "Machine Learning Engineer":  "machine learning engineer ML",
    "AI Engineer":                "AI engineer artificial intelligence",
    "DevOps Engineer":            "devops engineer CI/CD",
    "Cloud Engineer":             "cloud engineer AWS Azure GCP",
    "Frontend Developer":         "frontend developer react javascript",
    "Backend Developer":          "backend developer python java node",
    "Full Stack Developer":        "full stack developer",
    "Mobile Developer":           "mobile developer android ios flutter",
    "Cybersecurity Analyst":      "cybersecurity security analyst",
    "QA Engineer":                "QA quality assurance test engineer",
    "UI/UX Designer":             "UI UX designer product designer",
    "Database Administrator":     "database administrator DBA SQL",
    "Network Engineer":           "network engineer infrastructure",
    "Embedded Systems Engineer":  "embedded systems firmware engineer C++",
    "Business Analyst":           "business analyst requirements",
    "Data Engineer":              "data engineer ETL pipeline spark",
    "Site Reliability Engineer":  "SRE site reliability engineer",
    "Product Manager":            "product manager technology",
    "Scrum Master":               "scrum master agile project manager",
    "IT Security Manager":        "IT security manager CISO",
    "Cloud Architect":            "cloud architect solutions architect",
    "Blockchain Developer":       "blockchain developer web3 solidity",
    "Game Developer":             "game developer unity unreal engine",
    "IoT Engineer":               "IoT internet of things embedded",
    "IT Consultant":              "IT consultant technology advisor",
    "Systems Analyst":            "systems analyst IT analyst",
    "Data Modeling Engineer":     "data modeling warehouse engineer",
}

# ── Shared prediction kwargs (passed to model/predict.py) ────────────────────
# Bundled here so every route uses identical settings without repetition
PREDICT_KWARGS = dict(
    ensemble      = ensemble,
    preprocessor  = preprocessor,
    label_encoder = label_encoder,
    feature_order = feature_order,
    drop_corr     = drop_corr,
    role_to_domain= role_to_domain,
)

# ── Database ──────────────────────────────────────────────────────────────────
DB_PATH = Path("career_app.db")

def _init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, name TEXT DEFAULT 'Anonymous',
        top_role TEXT, domain TEXT, confidence REAL,
        top3 TEXT, profile TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS profiles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT, name TEXT, data TEXT, result TEXT)""")
    con.execute("""CREATE TABLE IF NOT EXISTS progress_snapshots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ts TEXT NOT NULL,
        name TEXT NOT NULL,
        top_role TEXT, domain TEXT, confidence REAL,
        readiness_score REAL, readiness_band TEXT,
        skill_gap_pct REAL,
        skills TEXT,
        top3 TEXT)""")
    con.commit()
    con.close()

_init_db()


def _log_prediction(name: str, results: list, profile: dict):
    top = results[0]
    con = sqlite3.connect(DB_PATH)
    con.execute(
        "INSERT INTO predictions(ts,name,top_role,domain,confidence,top3,profile) "
        "VALUES(?,?,?,?,?,?,?)",
        (datetime.now().isoformat(), name or "Anonymous",
         top["role"], top["domain"], top["probability"],
         json.dumps([r["role"] for r in results]),
         json.dumps(profile))
    )
    con.commit()
    con.close()


# ── Skill gap, peer compare, readiness, SHAP ─────────────────────────────────
SGAP_FIELDS = [
    "programming_skill","data_analytics_skill","data_science_ml_skill",
    "cloud_devops_skill","cybersecurity_skill","ui_ux_design_skill",
    "qa_testing_skill","web_dev_skill","mobile_dev_skill","db_sql_skill",
    "system_design_score","data_modeling_skill","api_design_skill",
    "embedded_c_cpp_skill","siem_experience_score","cgpa",
    "github_commits_90d","project_count","certifications_total",
    "internship_experience_count",
]
SGAP_LABELS = {
    "programming_skill":"Programming","data_analytics_skill":"Data Analytics",
    "data_science_ml_skill":"Data Science/ML","cloud_devops_skill":"Cloud/DevOps",
    "cybersecurity_skill":"Cybersecurity","ui_ux_design_skill":"UI/UX Design",
    "qa_testing_skill":"QA/Testing","web_dev_skill":"Web Dev",
    "mobile_dev_skill":"Mobile Dev","db_sql_skill":"Database/SQL",
    "system_design_score":"System Design","data_modeling_skill":"Data Modeling",
    "api_design_skill":"API Design","embedded_c_cpp_skill":"Embedded C/C++",
    "siem_experience_score":"SIEM Experience","cgpa":"CGPA",
    "github_commits_90d":"GitHub Commits","project_count":"Projects",
    "certifications_total":"Certifications","internship_experience_count":"Internships",
}
SGAP_SCALE = {
    "cgpa":10.0,"github_commits_90d":30.0,"project_count":2.0,
    "certifications_total":2.0,"internship_experience_count":2.0,
}

def _skill_gap(user: dict, role: str) -> tuple[list, float]:
    profile = role_profiles.get(role, {})
    gaps = []
    for f in SGAP_FIELDS:
        if f not in profile:
            continue
        u = min(float(user.get(f, 0)) * SGAP_SCALE.get(f, 1.0), 10)
        i = min(float(profile[f])    * SGAP_SCALE.get(f, 1.0), 10)
        gaps.append({"field": f, "label": SGAP_LABELS.get(f, f),
                     "user": round(u, 1), "ideal": round(i, 1),
                     "gap": round(i - u, 2)})
    gaps.sort(key=lambda x: -x["gap"])
    tot_u = sum(g["user"]  for g in gaps)
    tot_i = sum(g["ideal"] for g in gaps)
    pct   = round(min(tot_u / max(tot_i, 1) * 100, 100), 1)
    return gaps[:12], pct


def _peer_compare(user: dict, role: str) -> list:
    pcts = role_pctiles.get(role, {})
    out  = []
    for f, label in [
        ("cgpa","CGPA"),("programming_skill","Programming"),
        ("data_analytics_skill","Data Analytics"),("data_science_ml_skill","DS/ML"),
        ("cloud_devops_skill","Cloud/DevOps"),("cybersecurity_skill","Cybersecurity"),
        ("project_count","Projects"),("github_commits_90d","GitHub Commits"),
        ("certifications_total","Certifications"),
    ]:
        if f not in pcts:
            continue
        uv = float(user.get(f, 0))
        p  = pcts[f]
        rank, rc = (
            ("Top 25%","#10b981") if uv >= p["p75"] else
            ("Top 50%","#3b82f6") if uv >= p["p50"] else
            ("Top 75%","#f59e0b") if uv >= p["p25"] else
            ("Bottom 25%","#ef4444")
        )
        out.append({"field":f,"label":label,"user":uv,
                    "p25":p["p25"],"p50":p["p50"],"p75":p["p75"],
                    "mean":p["mean"],"rank":rank,"rank_color":rc})
    return out


def _readiness_score(user: dict, skill_gap_pct: float, peers: list, confidence: float) -> dict:
    skill_pts = round(skill_gap_pct * 0.40, 1)
    rank_map  = {"Top 25%":100,"Top 50%":75,"Top 75%":50,"Bottom 25%":25}
    peer_avg  = (sum(rank_map.get(p["rank"],"Bottom 25%") if isinstance(p.get("rank"), str)
                    else rank_map.get(p.get("rank","Bottom 25%"),25) for p in peers)
                 / max(len(peers), 1))
    peer_pts  = round(peer_avg * 0.25, 1)
    cgpa_norm = min(float(user.get("cgpa", 0)) / 10.0, 1.0)
    math_norm = min(float(user.get("math_scores", 0)) / 100.0, 1.0)
    cs_norm   = min(float(user.get("cs_fundamentals_scores", 0)) / 100.0, 1.0)
    acad_pts  = round((cgpa_norm*0.5 + math_norm*0.25 + cs_norm*0.25) * 100 * 0.20, 1)
    proj_s    = min(float(user.get("project_count", 0)) / 12.0, 1.0)
    intern_s  = min(float(user.get("internship_experience_count", 0)) / 3.0, 1.0)
    cert_s    = min(float(user.get("certifications_total", 0)) / 5.0, 1.0)
    gh_s      = min(float(user.get("github_commits_90d", 0)) / 250.0, 1.0)
    exp_pts   = round((proj_s*0.35 + intern_s*0.30 + cert_s*0.20 + gh_s*0.15) * 100 * 0.15, 1)
    total     = min(max(round(skill_pts + peer_pts + acad_pts + exp_pts, 1), 0), 100)
    band, bc  = (("Standout","#10b981") if total >= 85 else
                 ("Job-Ready","#3b82f6") if total >= 70 else
                 ("Developing","#f59e0b") if total >= 50 else
                 ("Not Ready","#ef4444"))
    return {"score":total,"band":band,"band_color":bc,
            "breakdown":{
                "skill_match":{"pts":skill_pts,"max":40,"label":"Skill match"},
                "peer_rank":  {"pts":peer_pts, "max":25,"label":"Peer ranking"},
                "academic":   {"pts":acad_pts, "max":20,"label":"Academic"},
                "experience": {"pts":exp_pts,  "max":15,"label":"Experience"},
            }}


def _shap_fallback(top_role: str) -> list:
    """
    When the live SHAP explainer is not ready, return global feature-importance
    values as a stand-in.  Marked with ``is_global=True`` so the UI can note
    that these are model-level (not personalised) importances.
    """
    if not feature_imp:
        return []
    # feature_imp is {feature_name: importance_score}
    items = sorted(feature_imp.items(), key=lambda x: -abs(float(x[1])))[:10]
    return [
        {
            "feature":   f,
            "label":     f.replace("_", " ").title(),
            "shap":      round(float(v), 4),
            "direction": "positive" if float(v) >= 0 else "negative",
            "is_global": True,
        }
        for f, v in items
    ]


def _shap_explanation(raw_input: dict, top_role: str) -> list:
    """
    Compute personalised per-prediction SHAP values for ``top_role``.

    Falls back to global feature importance if:
      • The explainer hasn't finished loading yet, OR
      • Another SHAP computation is in progress (semaphore busy), OR
      • The computation raises any exception.
    """
    if not shap_explainer:
        logger.debug("SHAP explainer not ready — using global fallback")
        return _shap_fallback(top_role)

    # Limit to 1 concurrent SHAP computation (the model is ~1 GB in RAM)
    acquired = _shap_semaphore.acquire(timeout=10)
    if not acquired:
        logger.warning("SHAP semaphore busy — returning global fallback")
        return _shap_fallback(top_role)

    try:
        from model.preprocess import align_and_preprocess
        X  = align_and_preprocess(raw_input, feature_order, drop_corr,
                                   preprocessor, log_shapes=False)
        sv = shap_explainer.shap_values(X)
        ci = list(label_encoder.classes_).index(top_role)

        # TreeExplainer returns different formats depending on SHAP version:
        #   - List of arrays : [class0_arr, class1_arr, …] each (n_samples, n_features)
        #   - 3-D numpy array: (n_samples, n_features, n_classes)
        #   - SHAP Explanation object with .values attribute
        if isinstance(sv, list):
            cs = sv[ci][0]
        elif hasattr(sv, "values"):
            cs = sv.values[0, :, ci] if sv.values.ndim == 3 else sv.values[0]
        else:
            cs = sv[0, :, ci]

        n_feats = min(len(cs), len(feature_order))
        seen, out = set(), []
        # top-6 positive + top-6 negative indices
        for idx in (list(np.argsort(cs[:n_feats])[::-1][:6]) +
                    list(np.argsort(cs[:n_feats])[:6])):
            if idx in seen or idx >= len(feature_order):
                continue
            seen.add(idx)
            f = feature_order[idx]
            v = float(cs[idx])
            out.append({
                "feature":   f,
                "label":     f.replace("_", " ").title(),
                "shap":      round(v, 4),
                "direction": "positive" if v > 0 else "negative",
                "is_global": False,
            })
        out.sort(key=lambda x: -abs(x["shap"]))
        return out[:10]

    except Exception as exc:
        logger.warning("SHAP live computation failed: %s — using fallback", exc)
        return _shap_fallback(top_role)
    finally:
        _shap_semaphore.release()


def _action_plan(skill_gaps: list, user_input: dict) -> list:
    plan, seen = [], set()
    for gap in sorted(skill_gaps, key=lambda g: g["gap"], reverse=True):
        f = gap["field"]
        if f in action_res and f not in seen:
            r = action_res[f]
            plan.append({"skill":    gap["label"], "gap": gap["gap"],
                         "title":    r["title"],   "resource": r["resource"],
                         "url":      r["url"],     "time": r["time"],
                         "tag":      r["tag"]})
            seen.add(f)
        if len(plan) == 3:
            break
    return plan


def _plain_explanation(user: dict, top_role: str, confidence: float,
                       readiness: dict, gaps: list) -> dict:
    """
    Generate 4-5 plain-English sentences summarising why the model chose
    this role and what the student should do next.
    """
    sentences = []

    # 1 — Confidence opener
    conf = round(confidence)
    if conf >= 70:
        sentences.append(
            f"The model is highly confident ({conf}%) that {top_role} "
            f"aligns with your skills, interests, and experience."
        )
    elif conf >= 50:
        sentences.append(
            f"With {conf}% confidence, {top_role} emerges as your best "
            f"career match based on your current profile."
        )
    else:
        sentences.append(
            f"Your profile shows some alignment with {top_role} ({conf}%) — "
            f"expanding your core skills would significantly strengthen this match."
        )

    # 2 — Strongest skill
    SKILL_NAMES = {
        "programming_skill":      "programming",
        "data_science_ml_skill":  "data science / ML",
        "cloud_devops_skill":     "cloud / DevOps",
        "cybersecurity_skill":    "cybersecurity",
        "web_dev_skill":          "web development",
        "ui_ux_design_skill":     "UI/UX design",
        "mobile_dev_skill":       "mobile development",
        "data_analytics_skill":   "data analytics",
        "db_sql_skill":           "database / SQL",
        "system_design_score":    "system design",
    }
    best_f = max(SKILL_NAMES, key=lambda f: float(user.get(f, 0)))
    best_v = float(user.get(best_f, 0))
    if best_v > 0:
        sentences.append(
            f"Your strongest area is {SKILL_NAMES[best_f]} ({best_v:.0f}/10), "
            f"which directly maps to the core competencies expected of a {top_role}."
        )

    # 3 — Experience summary
    projects   = int(user.get("project_count", 0))
    internships= int(user.get("internship_experience_count", 0))
    certs      = int(user.get("certifications_total", 0))
    commits    = int(user.get("github_commits_90d", 0))
    exp_parts  = []
    if projects:    exp_parts.append(f"{projects} project{'s' if projects!=1 else ''}")
    if internships: exp_parts.append(f"{internships} internship{'s' if internships!=1 else ''}")
    if certs:       exp_parts.append(f"{certs} certification{'s' if certs!=1 else ''}")
    if commits:     exp_parts.append(f"{commits} GitHub commits in 90 days")
    if exp_parts:
        sentences.append(
            f"Your hands-on portfolio — {', '.join(exp_parts)} — "
            f"gives you a practical edge when competing for {top_role} roles."
        )

    # 4 — Biggest gap (actionable)
    if gaps:
        g = gaps[0]
        sentences.append(
            f"Your biggest growth area is {g['label']} "
            f"(you: {g['user']:.1f}/10, role average: {g['ideal']:.1f}/10) — "
            f"closing this gap would push your readiness score up noticeably."
        )

    # 5 — Band-specific closing advice
    band = readiness.get("band", "Developing")
    sentences.append({
        "Standout":   "You are in the top tier of candidates — start applying and interview immediately.",
        "Job-Ready":  "You are job-ready. Build one standout portfolio project to differentiate yourself.",
        "Developing": "You are on the right track — keep building consistently and revisit your roadmap weekly.",
        "Not Ready":  "Focus on the fundamentals first — the Roadmap tab shows exactly where to invest your time.",
    }.get(band, "Keep building your skills steadily."))

    return {"sentences": sentences}


# ─── LLM Hybrid Predictor ────────────────────────────────────────────────────

def _profile_hash(user: dict) -> str:
    """Stable short hash of key numeric features — used as LLM cache key."""
    import hashlib
    KEY_FIELDS = [
        "programming_skill", "data_science_ml_skill", "cloud_devops_skill",
        "cybersecurity_skill", "web_dev_skill", "ui_ux_design_skill",
        "mobile_dev_skill", "data_analytics_skill", "db_sql_skill",
        "system_design_score", "embedded_c_cpp_skill", "qa_testing_skill",
        "cgpa", "project_count", "github_commits_90d", "certifications_total",
        "internship_experience_count", "interest_dev_overall",
        "interest_data_overall", "interest_cybersecurity",
    ]
    vals = ";".join(f"{f}={round(float(user.get(f, 0)), 1)}" for f in KEY_FIELDS)
    return hashlib.md5(vals.encode()).hexdigest()[:16]


def _build_llm_predict_prompt(user: dict) -> str:
    """Build a tight structured-output prompt for Claude career prediction."""
    valid_roles = sorted(label_encoder.classes_.tolist())

    # ── Top skills (non-zero, sorted descending) ──────────────────────────────
    skill_map = {
        "Programming":    user.get("programming_skill", 0),
        "Data Science/ML": user.get("data_science_ml_skill", 0),
        "Web Dev":        user.get("web_dev_skill", 0),
        "Cloud/DevOps":   user.get("cloud_devops_skill", 0),
        "Cybersecurity":  user.get("cybersecurity_skill", 0),
        "UI/UX Design":   user.get("ui_ux_design_skill", 0),
        "Mobile Dev":     user.get("mobile_dev_skill", 0),
        "Data Analytics": user.get("data_analytics_skill", 0),
        "Database/SQL":   user.get("db_sql_skill", 0),
        "System Design":  user.get("system_design_score", 0),
        "API Design":     user.get("api_design_skill", 0),
        "QA/Testing":     user.get("qa_testing_skill", 0),
        "Embedded C/C++": user.get("embedded_c_cpp_skill", 0),
    }
    active_skills = sorted(
        [(k, float(v)) for k, v in skill_map.items() if float(v) > 0],
        key=lambda x: -x[1]
    )
    skills_str = ", ".join(f"{k}: {v}/10" for k, v in active_skills) or "none provided"

    # ── Domain interests ──────────────────────────────────────────────────────
    interest_map = {
        "Dev/Engineering": user.get("interest_dev_overall", 0),
        "Data & AI":       user.get("interest_data_overall", 0),
        "Cybersecurity":   user.get("interest_cybersecurity", 0),
        "Cloud/Infra":     user.get("interest_cloud_infra_overall", 0),
        "UI/UX":           user.get("interest_ui_ux_design", 0),
        "Business":        user.get("interest_business_and_management", 0),
    }
    active_interests = sorted(
        [(k, float(v)) for k, v in interest_map.items() if float(v) > 0],
        key=lambda x: -x[1]
    )
    interests_str = ", ".join(f"{k}: {v}/10" for k, v in active_interests) or "none provided"

    # ── Detected tools / languages ────────────────────────────────────────────
    TOOL_FLAGS = {
        "Python": "lang_python", "Java": "lang_java",
        "JavaScript": "lang_javascript", "C/C++": "lang_c_cpp",
        "SQL": "lang_sql", "React": "frontend_react",
        "Angular": "frontend_angular", "Node.js": "backend_node",
        "Django": "backend_django", "Spring": "backend_spring",
        "AWS": "cloud_aws", "Azure": "cloud_azure",
        "Docker": "devops_docker", "Kubernetes": "devops_kubernetes",
        "Terraform": "devops_terraform", "Spark": "data_tool_spark",
        "Kafka": "data_tool_kafka", "Flutter": "mobile_flutter",
        "Kotlin": "mobile_kotlin", "Selenium": "testing_tool_selenium",
    }
    tools = [name for name, key in TOOL_FLAGS.items() if int(user.get(key, 0)) == 1]
    tools_str = ", ".join(tools) or "none detected"

    # ── Experience summary ────────────────────────────────────────────────────
    exp_str = (
        f"CGPA {user.get('cgpa', 0)}/10, "
        f"{user.get('project_count', 0)} projects, "
        f"{user.get('internship_experience_count', 0)} internships, "
        f"{user.get('certifications_total', 0)} certifications, "
        f"{user.get('github_commits_90d', 0)} GitHub commits (90d)"
    )

    roles_list = "\n".join(f"- {r}" for r in valid_roles)

    return (
        "You are an expert career prediction engine for Indian B.Tech students.\n"
        "Given the profile below, choose the BEST-FIT role from the exact list provided.\n\n"
        f"TECHNICAL SKILLS (0-10): {skills_str}\n"
        f"DOMAIN INTERESTS (0-10): {interests_str}\n"
        f"TOOLS & LANGUAGES: {tools_str}\n"
        f"ACADEMIC & EXPERIENCE: {exp_str}\n\n"
        f"VALID ROLES — choose ONLY from this list:\n{roles_list}\n\n"
        "Reply with ONLY a valid JSON object (no markdown fences, no explanation):\n"
        '{"top_role": "<exact role>", '
        '"alternative1": "<exact role>", '
        '"alternative2": "<exact role>", '
        '"confidence": <integer 10-99>, '
        '"reasoning": "<one sentence>", '
        '"key_strengths": ["<strength1>", "<strength2>"], '
        '"key_gaps": ["<gap1>", "<gap2>"]}'
    )


def _llm_predict(user: dict) -> dict | None:
    """
    Ask Groq (Llama 3.3) to independently predict the student's best-fit role.
    Returns a parsed dict or None on any failure (so ML-only is the safe fallback).
    Results are cached for LLM_PRED_CACHE_TTL seconds by profile hash.
    """
    if not groq_client:
        return None

    h   = _profile_hash(user)
    now = time.time()
    if h in _llm_pred_cache:
        ts, cached = _llm_pred_cache[h]
        if now - ts < LLM_PRED_CACHE_TTL:
            logger.info("LLM predict: cache hit [%s]", h)
            return cached

    try:
        prompt = _build_llm_predict_prompt(user)
        logger.info("LLM predict: calling Groq/%s (prompt=%d chars)", GROQ_MODEL, len(prompt))

        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": "You are an expert career prediction engine. Respond ONLY with valid JSON, no markdown fences."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )
        raw = response.choices[0].message.content.strip()

        # Strip accidental markdown code fences
        if raw.startswith("```"):
            parts = raw.split("```")
            raw = parts[1].lstrip("json").strip() if len(parts) > 1 else raw

        result = json.loads(raw)

        # ── Validate top_role is a known class ────────────────────────────────
        valid_set = set(label_encoder.classes_.tolist())
        top = result.get("top_role", "")
        if top not in valid_set:
            # Fuzzy fallback: partial string match
            match = next(
                (r for r in valid_set
                 if top.lower() in r.lower() or r.lower() in top.lower()), None
            )
            if match:
                logger.info("LLM predict: fuzzy-matched '%s' → '%s'", top, match)
                result["top_role"] = match
            else:
                logger.warning("LLM predict: unknown role '%s' — discarding", top)
                return None

        # Validate alternatives (silently nullify bad ones)
        for key in ("alternative1", "alternative2"):
            if result.get(key) not in valid_set:
                result[key] = None

        # Clamp confidence to [10, 99]
        result["confidence"] = max(10, min(99, int(result.get("confidence", 70))))

        _llm_pred_cache[h] = (now, result)
        logger.info("LLM predict: '%s' conf=%d%%", result["top_role"], result["confidence"])
        return result

    except json.JSONDecodeError as exc:
        logger.warning("LLM predict: JSON parse failed — %s | raw=%r", exc, raw[:120])
        return None
    except Exception as exc:
        logger.warning("LLM predict: failed — %s", exc)
        return None


def _blend_predictions(
    ml_results: list,
    llm_result: dict | None,
) -> tuple[list, str, dict]:
    """
    Merge ML and LLM predictions into a single authoritative ranked list.

    Agreement logic:
      VERIFIED  — LLM top == ML #1  → boost ML confidence +8 %, add verified badge
      PARTIAL   — LLM top in ML #2/#3 → boost that role +12 %, flag partial
      SPLIT     — LLM top not in ML top-3 → append LLM role as AI suggestion
      ML_ONLY   — No LLM result → return ML unchanged

    Returns:
      blended_results  — final ranked list (may have 4 entries on split)
      agreement        — 'verified' | 'partial' | 'split' | 'ml_only'
      blend_meta       — extra LLM context dict for the response JSON
    """
    if not llm_result:
        return ml_results, "ml_only", {}

    llm_top  = llm_result.get("top_role")
    llm_conf = llm_result.get("confidence", 70)
    ml_roles = [r["role"] for r in ml_results]

    blend_meta = {
        "llm_role":       llm_top,
        "llm_confidence": llm_conf,
        "llm_reasoning":  llm_result.get("reasoning", ""),
        "llm_strengths":  llm_result.get("key_strengths", []),
        "llm_gaps":       llm_result.get("key_gaps", []),
        "llm_alt1":       llm_result.get("alternative1"),
        "llm_alt2":       llm_result.get("alternative2"),
    }

    blended = [r.copy() for r in ml_results]

    if llm_top == ml_roles[0]:
        # Both agree on #1 — strong signal
        blended[0]["probability"] = round(
            min(blended[0]["probability"] * 1.08, 99.0), 1)
        blended[0]["llm_verified"] = True
        agreement = "verified"
        logger.info("Blend: VERIFIED — both agree on '%s'", llm_top)

    elif llm_top in ml_roles[1:]:
        # LLM agrees within top-3 — pull that role up a bit
        idx = ml_roles.index(llm_top)
        blended[idx]["probability"] = round(
            blended[idx]["probability"] * 1.12, 1)
        blended[idx]["llm_partial"] = True
        agreement = "partial"
        logger.info("Blend: PARTIAL — LLM agrees with ML #%d '%s'", idx + 1, llm_top)

    else:
        # LLM sees something the ML model doesn't — add it as an AI suggestion
        llm_entry = {
            "role":        llm_top,
            "domain":      role_to_domain.get(llm_top, "Technology"),
            "probability": round(llm_conf * 0.55, 1),  # scaled down for display consistency
            "llm_only":    True,
            "note":        "AI suggests this role based on real-world career patterns",
        }
        blended.append(llm_entry)
        agreement = "split"
        logger.info("Blend: SPLIT — LLM suggests '%s' (not in ML top-3)", llm_top)

    return blended, agreement, blend_meta



# ─── Resume text parser (unchanged — well-tested) ────────────────────────────
def _parse_resume_text(text: str) -> dict:
    """Extract model features from raw resume text."""
    t  = text.lower()
    ex: dict = {}

    # Binary tool flags
    SKILL_KW = {
        "lang_python":["python"],"lang_java":["java ","java\n","java,"],
        "lang_javascript":["javascript","js ","node.js","nodejs","react","angular","vue"],
        "lang_c_cpp":["c++","c/c++","cpp"," c,"],"lang_sql":["sql","mysql","postgresql","sqlite"],
        "frontend_react":["react"],"frontend_angular":["angular"],
        "backend_node":["node.js","nodejs","express"],"backend_django":["django"],
        "backend_spring":["spring boot","springboot"],"db_postgres":["postgresql","postgres"],
        "data_tool_spark":["apache spark","pyspark","spark"],"data_tool_airflow":["airflow"],
        "data_tool_kafka":["kafka"],"cloud_aws":["aws","amazon web services"],
        "cloud_azure":["azure"],"devops_docker":["docker"],"devops_kubernetes":["kubernetes","k8s"],
        "devops_terraform":["terraform"],"observability_prometheus":["prometheus"],
        "observability_grafana":["grafana"],"security_tool_siem":["siem","splunk","qradar"],
        "security_tool_wireshark":["wireshark"],"security_tool_burpsuite":["burp suite","burpsuite"],
        "testing_tool_selenium":["selenium"],"testing_tool_jmeter":["jmeter"],
        "mobile_kotlin":["kotlin"],"mobile_flutter":["flutter"],
    }
    for f, kws in SKILL_KW.items():
        ex[f] = 1 if any(k in t for k in kws) else 0

    # Skill scores — multi-signal scoring
    _hits = lambda kws: sum(1 for k in kws if k in t)

    ex["programming_skill"]    = min(_hits(["python","java","c++","javascript","typescript","golang","rust","kotlin","dart","swift","programming","software development","coding","developer","engineer","sde","software engineer"]) * 1.5, 9.0)
    ex["data_analytics_skill"] = min(_hits(["data analytics","analytics","tableau","power bi","looker","data analysis","business intelligence","bi analyst"]) * 2.0, 9.0)
    ex["data_science_ml_skill"]= min(_hits(["machine learning","deep learning","tensorflow","pytorch","scikit","nlp","llm","neural","generative ai","mlops","ml engineer","data scientist","ai engineer"]) * 1.8, 9.5)
    ex["cloud_devops_skill"]   = min(_hits(["aws","azure","gcp","google cloud","cloud","devops","ci/cd","jenkins","github actions","pipeline","infrastructure as code"]) * 1.2, 9.0)
    ex["cybersecurity_skill"]  = min(_hits(["cybersecurity","security","penetration","ethical hacking","soc analyst","vulnerability","ctf","owasp","firewall","incident response"]) * 2.5, 9.0)
    ex["web_dev_skill"]        = min(_hits(["web development","frontend","html","css","react","angular","vue","next.js","nuxt","bootstrap","tailwind","web developer"]) * 1.5, 9.0)
    ex["mobile_dev_skill"]     = min(_hits(["mobile","android","ios","flutter","kotlin","swift","react native","xamarin"]) * 2.0, 9.0)
    ex["db_sql_skill"]         = min(_hits(["database","sql","mysql","postgresql","mongodb","redis","nosql","oracle","sqlite","db2"]) * 1.5, 9.0)
    ex["system_design_score"]  = min(_hits(["system design","architecture","distributed systems","microservices","kafka","grpc","scalab"]) * 2.0, 9.0)
    ex["networking_sysadmin_skill"] = min(_hits(["networking","linux","sysadmin","tcp/ip","dns","vpn","cisco","network engineer"]) * 2.0, 9.0)
    ex["embedded_c_cpp_skill"] = min(_hits(["embedded","firmware","microcontroller","rtos","arduino","raspberry pi","stm32","vhdl","fpga"]) * 2.5, 9.0)
    ex["ui_ux_design_skill"]   = min(_hits(["ui/ux","ux design","user experience","figma","adobe xd","wireframe","prototype","usability"]) * 2.5, 9.0)
    ex["qa_testing_skill"]     = min(_hits(["testing","quality assurance","qa","test automation","selenium","pytest","unit test","integration test","jmeter"]) * 2.0, 9.0)
    ex["business_analysis_skill"] = min(_hits(["business analysis","requirements","stakeholder","product","agile","scrum","jira","confluence","business analyst"]) * 2.0, 9.0)
    ex["api_design_skill"]     = min(_hits(["api","rest","graphql","fastapi","swagger","openapi","postman"]) * 1.5, 9.0)
    ex["data_modeling_skill"]  = min(_hits(["data modeling","data warehouse","star schema","snowflake","dbt","dimensional modeling","etl"]) * 2.5, 9.0)

    # CGPA
    for pat in [r'cgpa[\s:]*([0-9]+\.?[0-9]*)', r'gpa[\s:]*([0-9]+\.?[0-9]*)', r'([89]\.[0-9])\s*(?:cgpa|gpa|/10)']:
        m = re.search(pat, t)
        if m:
            try:
                v = float(m.group(1))
                if 0 < v <= 10:
                    ex["cgpa"] = v
                    break
            except Exception:
                pass

    # Project count
    for pat in [r'(\d+)\+?\s*projects?', r'projects?[\s:]*(\d+)']:
        m = re.search(pat, t)
        if m:
            try: ex["project_count"] = min(int(m.group(1)), 25); break
            except Exception: pass

    # GitHub commits
    for pat in [r'(\d+)\+?\s*commits?', r'commits?[\s:]*(\d+)']:
        m = re.search(pat, t)
        if m:
            try: ex["github_commits_90d"] = min(int(m.group(1)), 500); break
            except Exception: pass

    # Internships
    im = re.search(r'(\d+)\s+internships?', t)
    if im:
        ex['internship_experience_count'] = min(int(im.group(1)), 5)
    else:
        n = len(re.findall(r'intern', t))
        if n: ex['internship_experience_count'] = min(n, 5)

    # Certifications
    n = len(re.findall(r'certif|certified|comptia|cisco|oscp', t))
    if n: ex["certifications_total"] = min(n, 8)

    # Repos
    m = re.search(r'(\d+)\s*(?:public\s*)?repos?', t)
    if m:
        try: ex["github_repos_count"] = min(int(m.group(1)), 100)
        except Exception: pass

    # Soft skills
    if any(k in t for k in ["passionate","enthusiastic","motivated","driven","dedicated"]):
        ex.setdefault("learning_motivation", 8.0)
    if any(k in t for k in ["team","collaboration","cross-functional","led","managed"]):
        ex.setdefault("teamwork_behavior", 7.5)
    if any(k in t for k in ["communic","present","stakeholder","client"]):
        ex.setdefault("communication_skill", 7.5)

    # Infer interest from detected skills
    def _s2i(score): return round(min(score * 1.1, 10.0), 1)

    if ex.get('data_science_ml_skill', 0) > 2:
        ex['interest_data_overall'] = _s2i(ex['data_science_ml_skill'])
    elif ex.get('data_analytics_skill', 0) > 2:
        ex['interest_data_overall'] = _s2i(ex['data_analytics_skill'])

    backend_bonus = 2.0 if (ex.get('backend_spring',0) or ex.get('backend_django',0) or ex.get('backend_node',0)) else 0.0
    prog_sig = max(ex.get('programming_skill',0), ex.get('web_dev_skill',0)) + backend_bonus
    if prog_sig > 2:
        ex['interest_dev_overall'] = _s2i(min(prog_sig, 10))

    if ex.get('cloud_devops_skill', 0) > 2:
        ex['interest_cloud_infra_overall'] = _s2i(ex['cloud_devops_skill'])
    if ex.get('cybersecurity_skill', 0) > 2:
        ex['interest_cybersecurity'] = _s2i(ex['cybersecurity_skill'])
    if ex.get('ui_ux_design_skill', 0) > 2:
        ex['interest_ui_ux_design'] = _s2i(ex['ui_ux_design_skill'])
    elif ex.get('ui_ux_design_skill', 0) == 0 and ex.get('data_science_ml_skill', 0) > 3:
        ex['interest_ui_ux_design'] = 1.0

    # Infer project type counts
    total_proj = ex.get('project_count', 0)
    if total_proj > 0:
        if ex.get('data_science_ml_skill', 0) > 3:
            ex['projects_ml_ai'] = max(1, round(total_proj * min(ex['data_science_ml_skill']/10, 0.6)))
        if ex.get('data_analytics_skill', 0) > 3:
            ex['projects_data_analytics'] = max(1, round(total_proj * min(ex['data_analytics_skill']/10, 0.3)))
        if ex.get('web_dev_skill', 0) > 3:
            ex['projects_frontend'] = max(1, round(total_proj * min(ex['web_dev_skill']/10, 0.4)))
            ex['projects_fullstack'] = max(0, round(total_proj * 0.2))
        if ex.get('cloud_devops_skill', 0) > 3:
            ex['projects_cloud'] = max(1, round(total_proj * 0.2))
        if ex.get('cybersecurity_skill', 0) > 3:
            ex['projects_security_defense'] = max(1, round(total_proj * 0.3))
        has_backend = ex.get('backend_spring',0) or ex.get('backend_django',0) or ex.get('backend_node',0)
        if (has_backend or ex.get('api_design_skill',0) > 3) and ex.get('programming_skill',0) > 3:
            ex['projects_backend'] = max(1, round(total_proj * 0.5))

    # Infer cognitive from CGPA
    if ex.get('cgpa', 0) >= 8.5:
        ex.setdefault('cognitive_ability_score', 8.0)
        ex.setdefault('math_scores', 82)
        ex.setdefault('cs_fundamentals_scores', 80)
    elif ex.get('cgpa', 0) >= 7.5:
        ex.setdefault('cognitive_ability_score', 7.0)
        ex.setdefault('math_scores', 72)
        ex.setdefault('cs_fundamentals_scores', 72)

    return ex


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html", model_meta=model_meta, test_metrics=test_metrics)


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", model_meta=model_meta,
                           test_metrics=test_metrics, analytics=analytics_data,
                           feature_imp=feature_imp)


@app.route("/compare")
def compare_page():
    roles = sorted(label_encoder.classes_.tolist())
    return render_template("compare.html", roles=roles,
                           profiles=role_profiles, salaries=salary_data)


@app.route("/progress")
def progress_page():
    return render_template("progress.html", model_meta=model_meta,
                           test_metrics=test_metrics)


@app.route("/profiles")
def profiles_page():
    return render_template("profiles.html")


@app.route("/api/save_snapshot", methods=["POST"])
def save_snapshot():
    """Save a skill progress snapshot after prediction."""
    try:
        d = request.get_json(force=True) or {}
        name = (d.get("name", "") or "").strip()
        if not name:
            return jsonify({"success": False, "error": "Name required"}), 400

        top    = (d.get("predictions") or [{}])[0]
        rs     = d.get("readiness") or {}
        sg     = d.get("skill_gap") or {}
        skills = d.get("skills") or {}

        with sqlite3.connect(DB_PATH) as con:
            con.execute(
                "INSERT INTO progress_snapshots "
                "(ts, name, top_role, domain, confidence, "
                " readiness_score, readiness_band, skill_gap_pct, skills, top3) "
                "VALUES(?,?,?,?,?,?,?,?,?,?)",
                (datetime.now().isoformat(),
                 name,
                 top.get("role", ""),
                 top.get("domain", ""),
                 top.get("probability", 0),
                 rs.get("score", 0),
                 rs.get("band", ""),
                 sg.get("overall_pct", 0),
                 json.dumps(skills),
                 json.dumps([p.get("role", "") for p in d.get("predictions", [])]))
            )
        logger.info("Progress snapshot saved for '%s'", name)
        return jsonify({"success": True})
    except Exception as exc:
        logger.exception("save_snapshot error")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/progress")
def get_progress():
    """Get progress snapshots for a user by name."""
    try:
        name = request.args.get("name", "").strip()
        if not name:
            # Return all unique names for the name picker
            with sqlite3.connect(DB_PATH) as con:
                names = con.execute(
                    "SELECT DISTINCT name FROM progress_snapshots ORDER BY name"
                ).fetchall()
            return jsonify({"success": True, "names": [n[0] for n in names]})

        with sqlite3.connect(DB_PATH) as con:
            rows = con.execute(
                "SELECT id, ts, name, top_role, domain, confidence, "
                "readiness_score, readiness_band, skill_gap_pct, skills, top3 "
                "FROM progress_snapshots WHERE name = ? ORDER BY ts ASC",
                (name,)
            ).fetchall()

        snapshots = []
        for r in rows:
            skills = {}
            try:
                skills = json.loads(r[9]) if r[9] else {}
            except Exception:
                pass
            top3 = []
            try:
                top3 = json.loads(r[10]) if r[10] else []
            except Exception:
                pass
            snapshots.append({
                "id":              r[0],
                "ts":              r[1],
                "date":            r[1][:10] if r[1] else "",
                "time":            r[1][11:16] if r[1] and len(r[1]) > 11 else "",
                "name":            r[2],
                "top_role":        r[3] or "—",
                "domain":          r[4] or "—",
                "confidence":      round(r[5], 1) if r[5] else 0,
                "readiness_score": round(r[6], 1) if r[6] else 0,
                "readiness_band":  r[7] or "—",
                "skill_gap_pct":   round(r[8], 1) if r[8] else 0,
                "skills":          skills,
                "top3":            top3,
            })

        return jsonify({"success": True, "snapshots": snapshots, "count": len(snapshots)})
    except Exception as exc:
        logger.exception("get_progress error")
        return jsonify({"success": False, "error": str(exc)}), 500



@app.route("/predict_v2", methods=["POST"])
def predict_v2():
    """
    Main prediction endpoint.
    Pipeline:
      1. Receive JSON
      2. Validate + clean inputs
      3. predict_top3() → top-3 roles with probabilities
      4. Enrich (skill gap, peer compare, readiness, SHAP, salary, roadmap…)
      5. Log to DB
      6. Return full JSON
    """
    try:
        raw  = request.get_json(force=True) or {}
        name = raw.pop("student_name", "Anonymous")

        # Step 1 — Validate
        from utils.validators import validate_and_clean
        vr = validate_and_clean(raw)
        logger.info("predict_v2 | name='%s' signal_warnings=%d", name, len(vr.warnings))

        cleaned = vr.cleaned

        # Step 2 — ML Predict (full pipeline in model/predict.py)
        from model.predict import predict_top3
        ml_results = predict_top3(cleaned, **PREDICT_KWARGS)
        logger.info("ML predict | top=%s (%.1f%%)  #2=%s  #3=%s",
                    ml_results[0]["role"], ml_results[0]["probability"],
                    ml_results[1]["role"] if len(ml_results) > 1 else "-",
                    ml_results[2]["role"] if len(ml_results) > 2 else "-")

        # Step 3 — LLM independent prediction (Claude Haiku)
        llm_result = _llm_predict(cleaned)

        # Step 4 — Hybrid blend: reconcile ML + LLM
        results, agreement, blend_meta = _blend_predictions(ml_results, llm_result)
        top_role = results[0]["role"]

        # Determine prediction method label for UI
        method_label = {
            "verified": "ML + AI Verified ✦",
            "partial":  "ML + AI Partial Agreement",
            "split":    "ML + AI Split View — review both",
            "ml_only":  "ML Ensemble",
        }.get(agreement, "ML Ensemble")

        logger.info("Hybrid | agreement=%s final_top='%s' method='%s'",
                    agreement, top_role, method_label)

        # Step 5 — Enrichment
        gaps, pct = _skill_gap(cleaned, top_role)
        peers     = _peer_compare(cleaned, top_role)
        readiness = _readiness_score(cleaned, pct, peers, results[0]["probability"])
        shap_exp  = _shap_explanation(cleaned, top_role)
        roadmap   = roadmaps.get(top_role, {})
        salary    = salary_data.get(top_role, {})
        radar     = radar_profiles.get(top_role, {})
        companies = companies_data.get(top_role, [])
        action    = _action_plan(gaps, cleaned)

        # Step 6 — Log
        _log_prediction(name, results, cleaned)

        plain_exp = _plain_explanation(
            cleaned, top_role, results[0]["probability"], readiness, gaps
        )

        return jsonify({
            "success":            True,
            "predictions":        results,
            "top_role":           top_role,
            "skill_gap":          {"gaps": gaps, "overall_pct": pct},
            "peer_compare":       peers,
            "readiness":          readiness,
            "shap":               shap_exp,
            "roadmap":            roadmap,
            "salary":             salary,
            "radar":              radar,
            "companies":          companies,
            "action_plan":        action,
            "warnings":           vr.warnings,
            "plain_explanation":  plain_exp,
            # ── Hybrid intelligence fields ─────────────────────────────────
            "prediction_method":  method_label,
            "agreement":          agreement,
            "llm_insight":        blend_meta,
        })

    except Exception as exc:
        import traceback
        logger.exception("predict_v2 error")
        return jsonify({"success": False, "error": str(exc),
                        "trace": traceback.format_exc()}), 500


@app.route("/github_profile", methods=["POST"])
def github_profile():
    """
    Fetch a GitHub user's profile and extract model features.
    Uses utils/github_api.py which handles all error cases.
    """
    try:
        username = (request.get_json(force=True) or {}).get("username", "").strip()
        if not username:
            return jsonify({"success": False, "error": "Username required"}), 400

        from utils.github_api import fetch_github_profile
        result, err = fetch_github_profile(username)

        if err:
            logger.warning("GitHub fetch failed for @%s: %s", username, err)
            return jsonify({"success": False, "error": err}), 400

        logger.info("GitHub: @%s loaded — %d features extracted",
                    username, len(result["extracted"]))
        return jsonify({"success": True, **result})

    except Exception as exc:
        logger.exception("github_profile error")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/parse_resume", methods=["POST"])
def parse_resume():
    try:
        text = (request.get_json(force=True) or {}).get("text", "")
        if not text.strip():
            return jsonify({"success": False, "error": "Empty text"}), 400
        return jsonify({"success": True, "extracted": _parse_resume_text(text)})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


# ─── LinkedIn profile text parser ─────────────────────────────────────────────
def _parse_linkedin_text(text: str) -> dict:
    """
    Extract model features from pasted LinkedIn profile text.

    LinkedIn profiles have a distinct structure compared to resumes:
    - Section headers: About, Experience, Education, Skills, Certifications, etc.
    - Experience entries: "Title · Company · Duration"
    - Skills with endorsement counts
    - Education with degree and institution
    """
    t = text.lower()
    ex: dict = {}

    # ── Detect LinkedIn sections ──────────────────────────────────────────────
    sections = {}
    section_headers = [
        "about", "summary", "experience", "education", "skills",
        "certifications", "licenses & certifications", "licenses and certifications",
        "projects", "publications", "volunteer", "honors", "awards",
        "recommendations", "courses", "languages",
    ]
    lines = text.split("\n")
    current_section = "header"
    sections[current_section] = []
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        if low in section_headers or any(
            low == h or low.startswith(h + " ") for h in section_headers
        ):
            current_section = low.split()[0]  # Normalize to first word
            sections.setdefault(current_section, [])
        else:
            sections.setdefault(current_section, []).append(stripped)

    # Flatten section text for keyword searching
    about_text = " ".join(sections.get("about", []) + sections.get("summary", [])).lower()
    exp_text = " ".join(sections.get("experience", [])).lower()
    skills_text = " ".join(sections.get("skills", [])).lower()
    edu_text = " ".join(sections.get("education", [])).lower()
    cert_text = " ".join(sections.get("certifications", []) + sections.get("licenses", [])).lower()
    proj_text = " ".join(sections.get("projects", [])).lower()
    full_text = t  # Already lowered

    # ── Binary tool flags (same as resume parser) ─────────────────────────────
    SKILL_KW = {
        "lang_python": ["python"], "lang_java": ["java ", "java\n", "java,"],
        "lang_javascript": ["javascript", "js ", "node.js", "nodejs", "react", "angular", "vue"],
        "lang_c_cpp": ["c++", "c/c++", "cpp", " c,"], "lang_sql": ["sql", "mysql", "postgresql", "sqlite"],
        "frontend_react": ["react"], "frontend_angular": ["angular"],
        "backend_node": ["node.js", "nodejs", "express"], "backend_django": ["django"],
        "backend_spring": ["spring boot", "springboot"], "db_postgres": ["postgresql", "postgres"],
        "data_tool_spark": ["apache spark", "pyspark", "spark"], "data_tool_airflow": ["airflow"],
        "data_tool_kafka": ["kafka"], "cloud_aws": ["aws", "amazon web services"],
        "cloud_azure": ["azure"], "devops_docker": ["docker"], "devops_kubernetes": ["kubernetes", "k8s"],
        "devops_terraform": ["terraform"], "observability_prometheus": ["prometheus"],
        "observability_grafana": ["grafana"], "security_tool_siem": ["siem", "splunk", "qradar"],
        "security_tool_wireshark": ["wireshark"], "security_tool_burpsuite": ["burp suite", "burpsuite"],
        "testing_tool_selenium": ["selenium"], "testing_tool_jmeter": ["jmeter"],
        "mobile_kotlin": ["kotlin"], "mobile_flutter": ["flutter"],
    }
    for f, kws in SKILL_KW.items():
        ex[f] = 1 if any(k in full_text for k in kws) else 0

    # ── Experience / Internship detection ─────────────────────────────────────
    # LinkedIn format: "Software Engineer · Google · 2 yrs 3 mos"
    exp_entries = sections.get("experience", [])
    intern_count = 0
    job_count = 0
    total_months = 0
    for line in exp_entries:
        low_line = line.lower()
        if "intern" in low_line:
            intern_count += 1
        elif any(w in low_line for w in ["engineer", "developer", "analyst", "designer",
                                          "manager", "consultant", "architect", "lead",
                                          "associate", "specialist", "administrator"]):
            job_count += 1
        # Parse duration: "2 yrs 3 mos", "1 yr", "6 mos"
        yr_m = re.search(r"(\d+)\s*(?:yr|year)s?", low_line)
        mo_m = re.search(r"(\d+)\s*(?:mo|month)s?", low_line)
        if yr_m:
            total_months += int(yr_m.group(1)) * 12
        if mo_m:
            total_months += int(mo_m.group(1))

    ex["internship_experience_count"] = max(intern_count, 1 if "intern" in full_text else 0)
    if ex["internship_experience_count"] == 0 and total_months > 0:
        ex["internship_experience_count"] = min(job_count, 5)

    # ── Certifications ────────────────────────────────────────────────────────
    cert_entries = sections.get("certifications", []) + sections.get("licenses", [])
    cert_count = 0
    for line in cert_entries:
        if line.strip() and len(line.strip()) > 3:
            # Skip section sub-headers and dates
            if not re.match(r"^(issued|expires|credential|see credential|show credential)", line.strip().lower()):
                cert_count += 1
    # Also count keywords
    cert_kw = len(re.findall(r"certif|certified|comptia|cisco|oscp|aws certified|azure certified|google certified", full_text))
    ex["certifications_total"] = max(cert_count // 2, cert_kw, min(cert_count, 8))  # LinkedIn often has multi-line per cert

    # ── Projects ──────────────────────────────────────────────────────────────
    proj_entries = [l for l in sections.get("projects", []) if l.strip() and len(l.strip()) > 5]
    proj_count = 0
    for line in proj_entries:
        if not re.match(r"^(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec|\d{4}|present|·|–|-)", line.strip().lower()):
            proj_count += 1
    proj_from_text = re.search(r"(\d+)\+?\s*projects?", full_text)
    if proj_from_text:
        proj_count = max(proj_count, int(proj_from_text.group(1)))
    ex["project_count"] = min(max(proj_count // 2, 1) if proj_count > 0 else 0, 25)

    # ── Education / CGPA ──────────────────────────────────────────────────────
    for pat in [r"cgpa[\s:]*([0-9]+\.?[0-9]*)", r"gpa[\s:]*([0-9]+\.?[0-9]*)",
                r"([89]\.[0-9])\s*(?:cgpa|gpa|/10)"]:
        m = re.search(pat, full_text)
        if m:
            try:
                v = float(m.group(1))
                if 0 < v <= 10:
                    ex["cgpa"] = v
                    break
            except Exception:
                pass
    # Infer CGPA from education level if not found
    if "cgpa" not in ex:
        if any(k in edu_text for k in ["iit", "nit", "bits", "iiit"]):
            ex["cgpa"] = 8.0
        elif any(k in edu_text for k in ["b.tech", "btech", "b.e.", "bachelor"]):
            ex["cgpa"] = 7.5

    # ── GitHub activity from LinkedIn text ────────────────────────────────────
    for pat in [r"(\d+)\+?\s*commits?", r"commits?[\s:]*(\d+)"]:
        m = re.search(pat, full_text)
        if m:
            try:
                ex["github_commits_90d"] = min(int(m.group(1)), 500)
                break
            except Exception:
                pass
    m = re.search(r"(\d+)\s*(?:public\s*)?repos?", full_text)
    if m:
        try:
            ex["github_repos_count"] = min(int(m.group(1)), 100)
        except Exception:
            pass

    # ── Skill scores — context-weighted ───────────────────────────────────────
    # LinkedIn skills section carries more weight; also check about + experience
    _hits = lambda kws, *texts: sum(1 for k in kws for txt in (texts or [full_text]) if k in txt)

    # Boost skills found in the dedicated Skills section
    def _score(kws, base_texts, max_val=9.0, multiplier=1.5):
        base = _hits(kws, *base_texts)
        skill_boost = _hits(kws, skills_text) * 0.5  # Extra weight for Skills section
        return min((base + skill_boost) * multiplier, max_val)

    ex["programming_skill"] = _score(
        ["python", "java", "c++", "javascript", "typescript", "golang", "rust", "kotlin",
         "dart", "swift", "programming", "software development", "coding", "developer",
         "engineer", "sde", "software engineer"],
        [full_text], 9.0, 1.5)
    ex["data_analytics_skill"] = _score(
        ["data analytics", "analytics", "tableau", "power bi", "looker", "data analysis",
         "business intelligence", "bi analyst"],
        [full_text], 9.0, 2.0)
    ex["data_science_ml_skill"] = _score(
        ["machine learning", "deep learning", "tensorflow", "pytorch", "scikit", "nlp",
         "llm", "neural", "generative ai", "mlops", "ml engineer", "data scientist", "ai engineer"],
        [full_text], 9.5, 1.8)
    ex["cloud_devops_skill"] = _score(
        ["aws", "azure", "gcp", "google cloud", "cloud", "devops", "ci/cd", "jenkins",
         "github actions", "pipeline", "infrastructure as code"],
        [full_text], 9.0, 1.2)
    ex["cybersecurity_skill"] = _score(
        ["cybersecurity", "security", "penetration", "ethical hacking", "soc analyst",
         "vulnerability", "ctf", "owasp", "firewall", "incident response"],
        [full_text], 9.0, 2.5)
    ex["web_dev_skill"] = _score(
        ["web development", "frontend", "html", "css", "react", "angular", "vue",
         "next.js", "nuxt", "bootstrap", "tailwind", "web developer"],
        [full_text], 9.0, 1.5)
    ex["mobile_dev_skill"] = _score(
        ["mobile", "android", "ios", "flutter", "kotlin", "swift", "react native", "xamarin"],
        [full_text], 9.0, 2.0)
    ex["db_sql_skill"] = _score(
        ["database", "sql", "mysql", "postgresql", "mongodb", "redis", "nosql", "oracle",
         "sqlite", "db2"],
        [full_text], 9.0, 1.5)
    ex["system_design_score"] = _score(
        ["system design", "architecture", "distributed systems", "microservices", "kafka",
         "grpc", "scalab"],
        [full_text], 9.0, 2.0)
    ex["networking_sysadmin_skill"] = _score(
        ["networking", "linux", "sysadmin", "tcp/ip", "dns", "vpn", "cisco", "network engineer"],
        [full_text], 9.0, 2.0)
    ex["embedded_c_cpp_skill"] = _score(
        ["embedded", "firmware", "microcontroller", "rtos", "arduino", "raspberry pi",
         "stm32", "vhdl", "fpga"],
        [full_text], 9.0, 2.5)
    ex["ui_ux_design_skill"] = _score(
        ["ui/ux", "ux design", "user experience", "figma", "adobe xd", "wireframe",
         "prototype", "usability"],
        [full_text], 9.0, 2.5)
    ex["qa_testing_skill"] = _score(
        ["testing", "quality assurance", "qa", "test automation", "selenium", "pytest",
         "unit test", "integration test", "jmeter"],
        [full_text], 9.0, 2.0)
    ex["business_analysis_skill"] = _score(
        ["business analysis", "requirements", "stakeholder", "product", "agile", "scrum",
         "jira", "confluence", "business analyst"],
        [full_text], 9.0, 2.0)
    ex["api_design_skill"] = _score(
        ["api", "rest", "graphql", "fastapi", "swagger", "openapi", "postman"],
        [full_text], 9.0, 1.5)
    ex["data_modeling_skill"] = _score(
        ["data modeling", "data warehouse", "star schema", "snowflake", "dbt",
         "dimensional modeling", "etl"],
        [full_text], 9.0, 2.5)
    ex["data_analytics_skill"] = _score(
        ["data analytics", "analytics", "tableau", "power bi", "looker", "data analysis",
         "business intelligence"],
        [full_text], 9.0, 2.0)

    # ── Soft skills ───────────────────────────────────────────────────────────
    if any(k in full_text for k in ["passionate", "enthusiastic", "motivated", "driven", "dedicated", "self-starter"]):
        ex.setdefault("learning_motivation", 8.0)
    if any(k in full_text for k in ["team", "collaboration", "cross-functional", "led", "managed", "leadership"]):
        ex.setdefault("teamwork_behavior", 7.5)
    if any(k in full_text for k in ["communic", "present", "stakeholder", "client", "public speaking"]):
        ex.setdefault("communication_skill", 7.5)

    # ── Infer interests from skills (same logic as resume parser) ─────────────
    def _s2i(score):
        return round(min(score * 1.1, 10.0), 1)

    if ex.get("data_science_ml_skill", 0) > 2:
        ex["interest_data_overall"] = _s2i(ex["data_science_ml_skill"])
    elif ex.get("data_analytics_skill", 0) > 2:
        ex["interest_data_overall"] = _s2i(ex["data_analytics_skill"])

    backend_bonus = 2.0 if (ex.get("backend_spring", 0) or ex.get("backend_django", 0) or ex.get("backend_node", 0)) else 0.0
    prog_sig = max(ex.get("programming_skill", 0), ex.get("web_dev_skill", 0)) + backend_bonus
    if prog_sig > 2:
        ex["interest_dev_overall"] = _s2i(min(prog_sig, 10))

    if ex.get("cloud_devops_skill", 0) > 2:
        ex["interest_cloud_infra_overall"] = _s2i(ex["cloud_devops_skill"])
    if ex.get("cybersecurity_skill", 0) > 2:
        ex["interest_cybersecurity"] = _s2i(ex["cybersecurity_skill"])
    if ex.get("ui_ux_design_skill", 0) > 2:
        ex["interest_ui_ux_design"] = _s2i(ex["ui_ux_design_skill"])

    # ── Infer project type counts ─────────────────────────────────────────────
    total_proj = ex.get("project_count", 0)
    if total_proj > 0:
        if ex.get("data_science_ml_skill", 0) > 3:
            ex["projects_ml_ai"] = max(1, round(total_proj * min(ex["data_science_ml_skill"] / 10, 0.6)))
        if ex.get("data_analytics_skill", 0) > 3:
            ex["projects_data_analytics"] = max(1, round(total_proj * min(ex["data_analytics_skill"] / 10, 0.3)))
        if ex.get("web_dev_skill", 0) > 3:
            ex["projects_frontend"] = max(1, round(total_proj * min(ex["web_dev_skill"] / 10, 0.4)))
            ex["projects_fullstack"] = max(0, round(total_proj * 0.2))
        if ex.get("cloud_devops_skill", 0) > 3:
            ex["projects_cloud"] = max(1, round(total_proj * 0.2))
        if ex.get("cybersecurity_skill", 0) > 3:
            ex["projects_security_defense"] = max(1, round(total_proj * 0.3))
        has_backend = ex.get("backend_spring", 0) or ex.get("backend_django", 0) or ex.get("backend_node", 0)
        if (has_backend or ex.get("api_design_skill", 0) > 3) and ex.get("programming_skill", 0) > 3:
            ex["projects_backend"] = max(1, round(total_proj * 0.5))

    # ── Infer cognitive from CGPA ─────────────────────────────────────────────
    if ex.get("cgpa", 0) >= 8.5:
        ex.setdefault("cognitive_ability_score", 8.0)
        ex.setdefault("math_scores", 82)
        ex.setdefault("cs_fundamentals_scores", 80)
    elif ex.get("cgpa", 0) >= 7.5:
        ex.setdefault("cognitive_ability_score", 7.0)
        ex.setdefault("math_scores", 72)
        ex.setdefault("cs_fundamentals_scores", 72)

    return ex


@app.route("/parse_linkedin", methods=["POST"])
def parse_linkedin():
    """Parse pasted LinkedIn profile text and extract model features."""
    try:
        text = (request.get_json(force=True) or {}).get("text", "")
        if not text.strip():
            return jsonify({"success": False, "error": "Empty text — paste your LinkedIn profile text."}), 400
        if len(text.strip()) < 50:
            return jsonify({"success": False, "error": "Text too short — paste your full LinkedIn profile (About, Experience, Skills, etc.)."}), 400

        extracted = _parse_linkedin_text(text)

        # Count detected signals for confidence feedback
        signals = sum(1 for v in extracted.values() if v and v != 0)

        logger.info("LinkedIn parse: %d features extracted from %d chars", signals, len(text))
        return jsonify({
            "success": True,
            "extracted": extracted,
            "signals": signals,
            "char_count": len(text),
        })
    except Exception as exc:
        logger.exception("LinkedIn parse error")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/parse_resume_pdf", methods=["POST"])
def parse_resume_pdf():
    """Upload a PDF resume, extract text with pdfplumber, parse to model features."""
    if not PDF_AVAILABLE:
        return jsonify({"success": False,
                        "error": "pdfplumber not installed. Run: pip install pdfplumber"}), 500
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file uploaded"}), 400

    f = request.files["file"]
    if not f.filename.lower().endswith(".pdf"):
        return jsonify({"success": False, "error": "Only PDF files supported"}), 400

    raw = f.read()
    if len(raw) > 5 * 1024 * 1024:
        return jsonify({"success": False, "error": "File too large (max 5 MB)"}), 400

    try:
        text_parts = []
        with pdfplumber.open(io.BytesIO(raw)) as pdf:
            for page in pdf.pages:
                # Try layout-aware extraction first, fall back to plain text
                t = page.extract_text(layout=True) or page.extract_text()
                if t:
                    text_parts.append(t)

        raw_text = "\n".join(text_parts)
        if not raw_text.strip():
            return jsonify({"success": False,
                            "error": "No text found — use a text-based PDF, not a scanned image."}), 400

        extracted = _parse_resume_text(raw_text)

        # ── Quality metrics for the UI ───────────────────────────────────────────
        skill_fields = [
            "programming_skill", "web_dev_skill", "data_science_ml_skill",
            "cloud_devops_skill", "cybersecurity_skill", "ui_ux_design_skill",
            "mobile_dev_skill", "data_analytics_skill", "db_sql_skill",
            "system_design_score", "api_design_skill", "qa_testing_skill",
        ]
        skills_found   = [(f, extracted[f]) for f in skill_fields if extracted.get(f, 0) > 0]
        tools_found    = [k for k in extracted if k.startswith(("lang_","frontend_","backend_",
                          "cloud_","devops_","data_tool_","mobile_","security_tool_",
                          "testing_tool_","observability_")) and extracted[k] == 1]
        acad_signals   = sum(1 for k in ["cgpa","project_count","internship_experience_count",
                                          "certifications_total","github_commits_90d"]
                              if extracted.get(k, 0) > 0)

        # Quality score 0-100
        q = min(100, round(
            len(skills_found) * 5 +
            len(tools_found)  * 3 +
            acad_signals      * 8 +
            (15 if extracted.get("cgpa", 0) > 0 else 0)
        ))

        logger.info("PDF parse: %d pages, %d chars, %d skills, %d tools, quality=%d",
                    len(text_parts), len(raw_text), len(skills_found), len(tools_found), q)

        return jsonify({
            "success":      True,
            "extracted":    extracted,
            "raw_text":     raw_text[:2000],
            "page_count":   len(text_parts),
            "char_count":   len(raw_text),
            # Quality metrics
            "quality_score": q,
            "skills_found":  [[f.replace("_"," ").title(), round(v, 1)] for f, v in skills_found],
            "tools_found":   [t.replace("lang_","").replace("_","+").replace("frontend ","")
                               .replace("backend ","").replace("devops ","")
                               .replace("data tool ","").replace("cloud ","")
                               .replace("mobile ","").replace("security tool ","")
                               .replace("testing tool ","").replace("observability ","")
                               .title() for t in tools_found],
            "acad_signals":  acad_signals,
        })
    except Exception as exc:
        logger.exception("PDF parse error")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/shap_status")
def shap_status():
    """Poll endpoint: returns whether live SHAP is ready."""
    return jsonify({
        "ready":   shap_explainer is not None,
        "loading": not _shap_loaded.is_set(),
    })


@app.route("/whatif", methods=["POST"])
def whatif():
    """What-if simulator: change one field, return new top-3."""
    try:
        data      = request.get_json(force=True) or {}
        field     = data.get("field")
        new_val   = float(data.get("value", 0))
        base_data = {k: v for k, v in data.items() if k not in ("field", "value")}
        base_data[field] = new_val

        from model.predict import predict_top3
        results = predict_top3(base_data, **PREDICT_KWARGS)
        return jsonify({"success": True, "predictions": results[:3],
                        "changed_field": field, "new_value": new_val})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/export_pdf", methods=["POST"])
def export_pdf():
    if not REPORTLAB_AVAILABLE:
        return jsonify({"success": False, "error": "reportlab not installed"}), 500
    try:
        d   = request.get_json(force=True) or {}
        buf = _build_pdf_report(
            d.get("student_name", "Student"), d.get("predictions", []),
            d.get("skill_gap", {}), d.get("peer_compare", []),
            d.get("roadmap", {}), d.get("shap", [])
        )
        if not buf:
            return jsonify({"success": False, "error": "PDF generation failed"}), 500
        name = (d.get("student_name","Report") or "Report").replace(" ","_")
        return send_file(buf, mimetype="application/pdf", as_attachment=True,
                         download_name=f"CareerAI_{name}.pdf")
    except Exception as exc:
        logger.exception("PDF export error")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/save_profile", methods=["POST"])
def save_profile():
    try:
        d    = request.get_json(force=True) or {}
        name = (d.get("name","") or "").strip()
        if not name:
            return jsonify({"success": False, "error": "Name required"}), 400
        con = sqlite3.connect(DB_PATH)
        con.execute("INSERT INTO profiles(ts,name,data,result) VALUES(?,?,?,?)",
                    (datetime.now().isoformat(), name,
                     json.dumps(d.get("profile",{})), json.dumps(d.get("result",{}))))
        con.commit(); con.close()
        return jsonify({"success": True})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/get_profiles")
def get_profiles():
    try:
        con  = sqlite3.connect(DB_PATH)
        rows = con.execute("SELECT id,ts,name,data,result FROM profiles "
                           "ORDER BY ts DESC LIMIT 20").fetchall()
        con.close()
        out = []
        for r in rows:
            res = json.loads(r[4]) if r[4] else {}
            out.append({"id":r[0],"ts":r[1][:16].replace("T"," "),"name":r[2],
                        "top_role":res.get("top_role","—"),
                        "confidence":res.get("confidence",0),
                        "profile":json.loads(r[3]),"result":res})
        return jsonify({"success": True, "profiles": out})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/analytics")
def get_analytics():
    return jsonify({"analytics": analytics_data, "feature_importance": feature_imp})


@app.route("/api/history")
def get_history():
    """Return recent predictions for the history timeline."""
    try:
        limit = min(int(request.args.get("limit", 30)), 100)
        con = sqlite3.connect(DB_PATH)
        rows = con.execute(
            "SELECT id, ts, name, top_role, domain, confidence, top3 "
            "FROM predictions ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
        total = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        con.close()
        history = []
        for r in rows:
            top3 = []
            try:
                top3 = json.loads(r[6]) if r[6] else []
            except Exception:
                pass
            history.append({
                "id": r[0],
                "ts": r[1][:16].replace("T", " ") if r[1] else "",
                "name": r[2] or "Anonymous",
                "top_role": r[3] or "—",
                "domain": r[4] or "—",
                "confidence": round(r[5], 1) if r[5] else 0,
                "top3": top3,
            })
        return jsonify({"success": True, "history": history, "total": total})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/salary", methods=["GET","POST"])
def get_salary():
    role = (request.get_json(force=True) or {}).get("role","") if request.method=="POST" \
           else request.args.get("role","")
    data = salary_data.get(role)
    if not data:
        for k in salary_data:
            if role.lower() in k.lower() or k.lower() in role.lower():
                data = salary_data[k]; break
    if data:
        return jsonify({"success": True, "salary": data, "role": role})
    return jsonify({"success": False, "error": "Role not found"}), 404


@app.route("/api/jobs")
def get_jobs():
    """Fetch live remote job listings from Remotive (free, no API key required)."""
    import urllib.request
    import urllib.parse

    role = request.args.get("role", "").strip()
    if not role:
        return jsonify({"success": False, "error": "Role required"}), 400

    # --- Cache hit? ---
    now = time.time()
    if role in _jobs_cache:
        ts, cached_jobs = _jobs_cache[role]
        if now - ts < JOBS_CACHE_TTL:
            logger.info("Jobs cache hit for '%s' (%d jobs)", role, len(cached_jobs))
            return jsonify({"success": True, "jobs": cached_jobs,
                            "count": len(cached_jobs), "cached": True})

    # --- Build search query ---
    query = ROLE_TO_SEARCH.get(role, role)
    encoded = urllib.parse.quote(query)
    url = f"https://remotive.com/api/remote-jobs?search={encoded}&limit=15"

    try:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "CareerAI/2.0 (B.Tech Final Year Project; Educational)"}
        )
        with urllib.request.urlopen(req, timeout=12) as resp:
            data = json.loads(resp.read().decode("utf-8"))

        raw_jobs = data.get("jobs", [])
        jobs = []
        for job in raw_jobs[:15]:
            pub  = (job.get("publication_date") or "")[:10]
            jtype = (job.get("job_type") or "full_time").replace("_", " ").title()
            tags  = [(t or "").strip() for t in (job.get("tags") or []) if (t or "").strip()][:5]
            loc   = (job.get("candidate_required_location") or "Remote / Worldwide")[:55]
            sal   = (job.get("salary") or "").strip()
            if len(sal) > 45:
                sal = sal[:45] + "…"

            jobs.append({
                "id":      job.get("id"),
                "title":   (job.get("title") or "")[:80],
                "company": job.get("company_name", "Unknown"),
                "logo":    job.get("company_logo", ""),
                "location": loc,
                "type":     jtype,
                "url":      job.get("url", "#"),
                "posted":   pub,
                "tags":     tags,
                "salary":   sal,
            })

        _jobs_cache[role] = (now, jobs)
        logger.info("Jobs fetched for '%s': %d results", role, len(jobs))
        return jsonify({"success": True, "jobs": jobs, "count": len(jobs), "source": "Remotive"})

    except Exception as exc:
        logger.warning("Jobs API failed for '%s': %s", role, exc)
        # Return gracefully so UI degrades cleanly
        return jsonify({"success": True, "jobs": [], "count": 0,
                        "note": "Could not fetch live jobs: " + str(exc)})


@app.route("/admin")
def admin():
    con  = sqlite3.connect(DB_PATH)
    rows = con.execute("SELECT ts,name,top_role,domain,confidence,top3 "
                       "FROM predictions ORDER BY ts DESC LIMIT 50").fetchall()
    total       = con.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
    domain_dist = con.execute("SELECT domain,COUNT(*) FROM predictions "
                              "GROUP BY domain ORDER BY 2 DESC").fetchall()
    top_roles   = con.execute("SELECT top_role,COUNT(*) FROM predictions "
                              "GROUP BY top_role ORDER BY 2 DESC LIMIT 10").fetchall()
    con.close()
    logs = [{"ts":r[0][:16].replace("T"," "),"name":r[1],"top_role":r[2],
             "domain":r[3],"confidence":r[4],"top3":json.loads(r[5]) if r[5] else []}
            for r in rows]
    return render_template("admin.html", logs=logs, total=total,
                           domain_dist=[{"domain":d[0],"count":d[1]} for d in domain_dist],
                           top_roles=[{"role":r[0],"count":r[1]} for r in top_roles])


# ─── PDF report builder ───────────────────────────────────────────────────────
def _build_pdf_report(name, results, skill_gap, peers, roadmap, shap_exp):
    if not REPORTLAB_AVAILABLE:
        return None
    buf = io.BytesIO()
    c   = rl_colors
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            topMargin=2*cm, bottomMargin=2*cm,
                            leftMargin=2*cm, rightMargin=2*cm)
    SS  = getSampleStyleSheet()
    H1  = ParagraphStyle("H1", parent=SS["Title"],   fontSize=20,
                          textColor=c.HexColor("#0d1117"), spaceAfter=4, alignment=TA_CENTER)
    H2  = ParagraphStyle("H2", parent=SS["Heading2"],fontSize=13,
                          textColor=c.HexColor("#4f8ef7"), spaceBefore=12, spaceAfter=4)
    NRM = ParagraphStyle("NRM",parent=SS["Normal"],  fontSize=10,
                          textColor=c.HexColor("#0d1117"), spaceAfter=3)
    MUT = ParagraphStyle("MUT",parent=SS["Normal"],  fontSize=9,
                          textColor=c.HexColor("#7d8997"), spaceAfter=2)
    ACCENT = c.HexColor("#4f8ef7"); GREEN = c.HexColor("#10b981"); WHITE = c.white
    LIGHT  = c.HexColor("#f0f4f8")

    def tbl(data, cols, hdr=ACCENT):
        t = Table(data, colWidths=cols)
        t.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),hdr),("TEXTCOLOR",(0,0),(-1,0),WHITE),
            ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),("FONTSIZE",(0,0),(-1,-1),9),
            ("GRID",(0,0),(-1,-1),0.4,c.HexColor("#d0d7e0")),
            ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE,LIGHT]),
            ("BOTTOMPADDING",(0,0),(-1,-1),5),("TOPPADDING",(0,0),(-1,-1),5),
        ]))
        return t

    story = [
        Paragraph("CareerAI — Career Prediction Report", H1),
        HRFlowable(width="100%", thickness=1, color=ACCENT), Spacer(1, 0.3*cm),
        tbl([["Student","Generated","Model","Dataset"],
             [name or "—", datetime.now().strftime("%d %b %Y %I:%M %p"),
              "Stacking Ensemble (LGBM+XGB+CatBoost+LR)",
              "30,000 records · 30 roles · 7 domains"]],
            [3.5*cm, 4.5*cm, 7*cm, 3*cm]),
        Spacer(1, 0.4*cm), Paragraph("Top Predictions", H2),
        tbl([["#","Role","Domain","Confidence"]] +
            [[f"#{i+1}", r["role"], r["domain"], f"{r['probability']}%"]
             for i, r in enumerate(results[:3])],
            [1*cm, 7*cm, 6*cm, 3*cm]),
    ]
    if shap_exp:
        story += [Spacer(1, 0.3*cm),
                  Paragraph(f"Why '{results[0]['role']}'? — SHAP", H2),
                  tbl([["Feature","SHAP","Direction"]] +
                      [[e["label"], f"{e['shap']:+.4f}",
                        "▲ Supports" if e["direction"]=="positive" else "▼ Reduces"]
                       for e in shap_exp[:8]],
                      [8*cm, 4*cm, 5*cm], c.HexColor("#a855f7"))]
    if skill_gap.get("gaps"):
        story += [Spacer(1, 0.3*cm),
                  Paragraph(f"Skill Gap (Match: {skill_gap.get('overall_pct',0)}%)", H2),
                  tbl([["Skill","You","Ideal","Gap"]] +
                      [[g["label"], f"{g['user']:.1f}/10",
                        f"{g['ideal']:.1f}/10", f"{g['gap']:+.1f}"]
                       for g in skill_gap["gaps"][:10]],
                      [7*cm, 3*cm, 3*cm, 4*cm], GREEN)]
    story += [Spacer(1, 0.5*cm),
              HRFlowable(width="100%", thickness=0.5, color=c.HexColor("#7d8997")),
              Paragraph("Generated by CareerAI · Stacking Ensemble · v2.0", MUT)]

    doc.build(story)
    buf.seek(0)
    return buf


# ─── LLM Career Advisor ──────────────────────────────────────────────────────
def _build_prompt(d: dict) -> str:
    top       = (d.get("predictions") or [{}])[0]
    role      = top.get("role", "Software Engineer")
    domain    = top.get("domain", "Tech")
    conf      = top.get("probability", 0)
    rs        = d.get("readiness", {})
    sal       = d.get("salary", {})
    sal_range = sal.get("india_lpa", [0, 0])
    gaps      = [g["label"] for g in (d.get("skill_gap") or {}).get("gaps", [])[:3] if g.get("gap", 0) > 0]
    alts      = [p["role"] for p in d.get("predictions", [])[1:3]]
    name      = d.get("student_name", "Student")
    top_sk    = d.get("top_skills", [])
    return (
        f"You are a warm, professional career counsellor for tech students in India.\n\n"
        f"Student: {name}\nPredicted role: {role}\nDomain: {domain}\n"
        f"Confidence: {conf}%\nReadiness: {rs.get('score',0)}/100 ({rs.get('band','')})\n"
        f"Salary (India): \u20b9{sal_range[0]}\u2013{sal_range[1]} LPA\n"
        f"Top skills: {', '.join(top_sk[:5]) or 'various'}\n"
        f"Key skill gaps: {', '.join(gaps) or 'minor areas'}\n"
        f"Alternative roles: {' and '.join(alts) or 'related roles'}\n\n"
        f"Write a personalised, encouraging career advice message (280\u2013320 words) with:\n"
        f"1. A warm opening (2-3 sentences) on why {role} fits them.\n"
        f"2. Their strengths based on skills & readiness.\n"
        f"3. Three concrete actions for the next 30 days to close gaps ({', '.join(gaps) or 'key skills'}).\n"
        f"4. A motivational closing sentence.\n"
        f"Use flowing paragraphs, no bullet points, no headers. Be specific to {role}."
    )


@app.route("/ai_advice", methods=["POST"])
def ai_advice():
    """Generate personalised career advice via Groq (Llama 3.3 70B)."""
    d    = request.get_json(force=True) or {}
    top  = (d.get("predictions") or [{}])[0]
    role = top.get("role", "your predicted role")
    name = d.get("student_name", "there")

    if not groq_client:
        fallback = (
            f"Hi {name}! Based on your profile, {role} is your strongest career match. "
            f"You've already built a solid foundation \u2014 now focus on deepening your core skills, "
            f"building a standout portfolio project, and earning a relevant industry certification. "
            f"Consistency is everything. Start today, and in 6 months you'll be interview-ready. "
            f"You've got this! \U0001f680"
        )
        return jsonify({"success": True, "advice": fallback, "source": "fallback"})

    try:
        prompt = _build_prompt(d)
        logger.info("Groq AI Advisor request | role=%s name=%s", role, name)
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a warm, expert career counsellor for Indian B.Tech students. "
                    "Always follow the exact section headers given in the prompt. "
                    "Be specific, encouraging, and data-driven. Never give generic advice."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.65,
            max_tokens=900,
        )
        advice = response.choices[0].message.content.strip()
        return jsonify({"success": True, "advice": advice, "source": "groq"})
    except Exception as exc:
        logger.exception("Groq AI Advisor error")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/ai_interview_prep", methods=["POST"])
def ai_interview_prep():
    """Generate interview prep tips for the predicted role."""
    d    = request.get_json(force=True) or {}
    top  = (d.get("predictions") or [{}])[0]
    role = top.get("role", "Software Engineer")
    name = d.get("student_name", "Student")
    gaps = [g["label"] for g in (d.get("skill_gap") or {}).get("gaps", [])[:3] if g.get("gap", 0) > 0]

    if not groq_client:
        return jsonify({"success": True, "content": f"Interview Prep for {role}:\n\n1. Review core concepts in your strongest technical areas.\n2. Practice 2-3 coding problems daily on LeetCode/HackerRank.\n3. Prepare STAR-format answers for behavioural questions.\n4. Research target companies and their tech stack.\n5. Mock interview with a peer at least twice before the real thing.", "source": "fallback"})

    prompt = (
        f"You are a top-tier tech interview coach specialising in Indian product companies (Flipkart, Razorpay, CRED, etc.) and MNCs (Google, Amazon, Microsoft).\n\n"
        f"=== CANDIDATE ===\n"
        f"Name: {name} | Target Role: {role}\n"
        f"Weak Areas to Address: {', '.join(gaps) or 'standard competencies'}\n\n"
        f"=== INTERVIEW PREP GUIDE FORMAT ===\n"
        f"Output EXACTLY these 4 sections with the emoji headers:\n\n"
        f"\U0001f9e0 Must-Master Technical Topics\n"
        f"List the top 5 technical topics for a {role} interview. For each, name ONE best free resource (YouTube channel, docs, or platform).\n\n"
        f"\U0001f4bb Practice Questions\n"
        f"Give 3 real-style interview questions (mix of coding/system design/domain). For each, include a 1-line hint on what interviewers look for.\n\n"
        f"\U0001f9d1 Behavioural Round Tips\n"
        f"Give 2 likely HR/behavioural questions for {role} with a STAR-method tip for each.\n\n"
        f"\U0001f4c5 7-Day Prep Sprint\n"
        f"A day-by-day plan (Day 1 through Day 7) with ONE specific focus task per day.\n\n"
        f"Rules: Be specific to {role} in Indian tech context. Total ~350 words. Use the exact headers above."
    )

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are an elite tech interview coach. Follow the exact section headers given. "
                    "Be hyper-specific, name real tools and companies, never give generic advice."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.55,
            max_tokens=1000,
        )
        content = response.choices[0].message.content.strip()
        return jsonify({"success": True, "content": content, "source": "groq"})
    except Exception as exc:
        logger.exception("Interview prep error")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/ai_learning_plan", methods=["POST"])
def ai_learning_plan():
    """Generate a 30-day learning plan for the predicted role."""
    d    = request.get_json(force=True) or {}
    top  = (d.get("predictions") or [{}])[0]
    role = top.get("role", "Software Engineer")
    name = d.get("student_name", "Student")
    gaps = [g["label"] for g in (d.get("skill_gap") or {}).get("gaps", [])[:5] if g.get("gap", 0) > 0]
    rs   = d.get("readiness", {})

    if not groq_client:
        return jsonify({"success": True, "content": f"30-Day Plan for {role}:\n\nWeek 1: Foundation — review core concepts\nWeek 2: Build a portfolio project\nWeek 3: Practice problems & mock interviews\nWeek 4: Certifications & networking", "source": "fallback"})

    readiness_score = rs.get('score', 0)
    readiness_band  = rs.get('band', 'Unknown')
    prompt = (
        f"You are a structured career development coach for Indian B.Tech students aiming for {role} roles.\n\n"
        f"=== STUDENT ===\n"
        f"Name: {name} | Readiness: {readiness_score}/100 ({readiness_band})\n"
        f"Priority Gaps: {', '.join(gaps) or 'general skill building'}\n\n"
        f"=== 30-DAY PLAN FORMAT ===\n"
        f"Output EXACTLY 4 week-blocks using these headers:\n\n"
        f"\U0001f4a1 Week 1 — Foundation & Gap Closing\n"
        f"Theme: [one sentence focus]. Daily tasks:\n"
        f"• Day 1-2: [specific task + resource/tool]\n"
        f"• Day 3-4: [specific task + resource/tool]\n"
        f"• Day 5-7: [specific task + resource/tool]\n"
        f"\u2705 Week 1 Milestone: [1 measurable deliverable]\n\n"
        f"\U0001f527 Week 2 — Build & Practice\n"
        f"[Same format]\n\n"
        f"\U0001f680 Week 3 — Projects & Portfolio\n"
        f"[Same format — must include one real deployable project]\n\n"
        f"\U0001f3c6 Week 4 — Interview Ready\n"
        f"[Same format — must include mock interview or community challenge]\n\n"
        f"Rules: All resources must be free and accessible in India. "
        f"Name specific YouTube channels, GitHub repos, or platforms (e.g. freeCodeCamp, Neetcode, CS50). "
        f"Total ~380 words. Be specific to {role}."
    )

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a structured career development coach. Follow the exact week-block format given. "
                    "Always name real, free, India-accessible resources. Be specific and measurable."
                )},
                {"role": "user", "content": prompt},
            ],
            temperature=0.55,
            max_tokens=1100,
        )
        content = response.choices[0].message.content.strip()
        return jsonify({"success": True, "content": content, "source": "groq"})
    except Exception as exc:
        logger.exception("Learning plan error")
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/ai_followup", methods=["POST"])
def ai_followup():
    """Handle follow-up career questions in a conversational way."""
    d    = request.get_json(force=True) or {}
    question = (d.get("question") or "").strip()
    if not question:
        return jsonify({"success": False, "error": "Question required"}), 400

    top  = (d.get("predictions") or [{}])[0]
    role = top.get("role", "Software Engineer")
    name = d.get("student_name", "Student")
    rs   = d.get("readiness", {})
    sal  = d.get("salary", {})

    if not groq_client:
        return jsonify({"success": True, "answer": f"Great question! For a {role} career path, I'd recommend focusing on building practical projects and networking with professionals in the field. Consider joining relevant communities on LinkedIn and GitHub.", "source": "fallback"})

    # Build conversation messages with context
    salary_range = sal.get('india_lpa', [0, 0])
    system_msg = (
        f"You are an expert career advisor for Indian tech students, laser-focused on {role} roles.\n"
        f"Student: {name} | Readiness: {rs.get('score', 0)}/100 ({rs.get('band', 'Unknown')}) "
        f"| Salary potential: \u20b9{salary_range[0]}\u2013{salary_range[1]} LPA.\n\n"
        f"RULES:\n"
        f"- Answer in 100-160 words maximum.\n"
        f"- Always give at least ONE specific, actionable tip (name a tool, platform, or resource).\n"
        f"- Be warm, direct, and confident — like a senior engineer mentoring a junior.\n"
        f"- Use plain text only. No bullet lists, no markdown, no asterisks.\n"
        f"- If asked about salaries, certifications, or companies, give India-specific data."
    )

    messages = [{"role": "system", "content": system_msg}]

    # Add chat history for context
    chat_history = d.get("chat_history", [])
    for msg in chat_history[-6:]:
        if msg.get("role") in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Ensure the current question is the last user message
    if not messages or messages[-1].get("content") != question:
        messages.append({"role": "user", "content": question})

    try:
        response = groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=messages,
            temperature=0.65,
            max_tokens=400,
        )
        answer = response.choices[0].message.content.strip()
        return jsonify({"success": True, "answer": answer, "source": "groq"})
    except Exception as exc:
        logger.exception("AI followup error")
        return jsonify({"success": False, "error": str(exc)}), 500


# ─── Entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(debug=True, port=5000)
