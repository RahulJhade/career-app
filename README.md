# 🎯 Career Prediction System
### B.Tech Final Year Capstone Project

> An end-to-end AI-powered system that predicts the best-fit IT career role for students based on academic performance, technical skills, interests, tools experience, and project portfolio.

---

## 📌 Table of Contents
- [Overview](#overview)
- [Demo](#demo)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Performance](#performance)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
- [Career Domains](#career-domains)
- [Tech Stack](#tech-stack)

---

## 🧠 Overview

The **Career Prediction System** helps final-year students identify the most suitable IT career path by analyzing 86 input features — including GPA, technical skill ratings, tool familiarity, certifications, and project experience — and producing a ranked list of the **Top-3 career roles** with confidence scores.

The system is built on a **Stacking Ensemble** of industry-grade gradient boosting models and served through a clean Flask web interface.

---

## 🚀 Demo


**https://career-app-fqr4.onrender.com**


Students fill out a profile form → the model processes 126 engineered features → Top-3 career roles are returned instantly with % confidence scores.

---

## 🏗️ Model Architecture

```
Input (86 raw features)
    │
    ├─► Feature Engineering  ──►  142 features
    ├─► Correlation Drop      ──►  126 features
    └─► Standard Scaling + Imputation
              │
    ┌─────────▼──────────────────────────────────┐
    │            Stacking Ensemble               │
    │                                            │
    │   ┌──────────┐  ┌─────────┐  ┌──────────┐ │
    │   │ LightGBM │  │ XGBoost │  │ CatBoost │ │
    │   └────┬─────┘  └────┬────┘  └────┬─────┘ │
    │        └─────────────┼────────────┘        │
    │                      ▼                     │
    │         Logistic Regression (Meta-Learner) │
    └──────────────────────┬─────────────────────┘
                           │
                           ▼
          Top-3 Career Role Predictions + Confidence %
```

### Why Stacking?
- **LightGBM** — fast training on tabular data, handles sparse features well
- **XGBoost** — strong baseline with regularization
- **CatBoost** — robust handling of categorical skill/tool features
- **Meta-Learner (LR)** — combines base model outputs for final calibrated predictions

---

## 📊 Dataset

| Property | Details |
|---|---|
| Total Records | 30,000 student profiles |
| Raw Features | 86 (academic, skills, interests, tools, experience) |
| Engineered Features | 142 → 126 after correlation pruning |
| Career Domains | 7 |
| Final Job Roles | 30 (merged from 40 original) |

---

## 📈 Performance

| Metric | Score |
|---|---|
| Test Accuracy | ~85% |
| Test Macro F1 | ~83% |
| **Test Top-3 Accuracy** | **~99%** |

> **Top-3 Accuracy of ~99%** means the correct career role appears in the top 3 predictions for virtually every student — making this highly reliable as a recommendation tool.

---

## 📁 Project Structure

```
career_app/
│
├── app.py                        # Flask web application & API routes
├── train_model.py                # Full training pipeline (EDA → Features → Model → Save)
├── career_dataset_final.csv      # Dataset (30,000 records)
├── requirements.txt              # Python dependencies
│
├── templates/
│   └── index.html                # Frontend UI
│
└── artifacts_layer/              # Serialized model artifacts (auto-generated)
    ├── stacking_ensemble.joblib      # Trained stacking ensemble
    ├── preprocess.joblib             # Scaler + imputer pipeline
    ├── label_encoder.joblib          # Target class encoder
    ├── feature_order.json            # Expected feature input order
    ├── role_to_domain.json           # Role → Domain mapping
    ├── role_merge_map.json           # Original 40 → 30 role merge map
    ├── dropped_correlated_features.json  # Features removed (ρ > threshold)
    └── model_meta.json               # Training metadata & version info
```

---

## ⚙️ Getting Started

### Prerequisites
- Python 3.8+
- pip

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/career-app.git
cd career-app
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Train the model
> Skip this step if the `artifacts_layer/` folder is already present.

```bash
python train_model.py
```

This will:
- Load and clean the dataset
- Engineer features and drop correlated ones
- Train the stacking ensemble
- Save all artifacts to `artifacts_layer/`

### Step 4 — Launch the web app
```bash
python app.py
```

Open your browser at: **http://localhost:5000**

---

## 🗂️ Career Domains

The system predicts across **7 career domains** covering **30 job roles**:

| # | Domain | Example Roles |
|---|---|---|
| 1 | 💻 Software Engineering | Backend, Frontend, Full-Stack, Mobile, Game Dev |
| 2 | 📊 Data & Artificial Intelligence | Data Science, ML Engineer, Data Engineering, BI |
| 3 | 🔐 Cybersecurity | SOC Analyst, Penetration Tester, Cloud Security, GRC |
| 4 | ☁️ Cloud, DevOps & Platform Eng. | DevOps Engineer, Cloud Architect, SRE |
| 5 | 🎨 UI/UX & Product | UX Designer, Product Manager |
| 6 | 🧪 Quality Assurance & Testing | QA Engineer, Automation, Performance Testing |
| 7 | 🖥️ Systems & Infrastructure | Embedded Systems, Database Admin, Network Engineer |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Models | LightGBM, XGBoost, CatBoost, Scikit-learn |
| Preprocessing | Pandas, NumPy, Scikit-learn Pipelines |
| Web Framework | Flask |
| Serialization | Joblib |
| Frontend | HTML / CSS / Jinja2 |

---

## 👨‍💻 Author

Built as a **B.Tech Final Year Capstone Project**.  
Feel free to fork, contribute, or raise issues!

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
