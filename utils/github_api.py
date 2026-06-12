"""
utils/github_api.py
─────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS:
  GitHub profile data gives real-world signals that forms can't fake:
  primary language usage, repo diversity, star counts.  These map directly
  to model features (lang_python, web_dev_skill, etc.).

FAILURE PREVENTION:
  1. Invalid username     → 404 → friendly error string (no exception raised)
  2. Rate limited         → 403 → informative message with fix instructions
  3. Network error        → caught, returns (None, error_str)
  4. No GITHUB_TOKEN set  → works but at 60 req/hour anonymous limit

GITHUB_TOKEN:
  Add to your environment to get 5000 req/hour:
    Windows:  $env:GITHUB_TOKEN="ghp_yourtoken"
    Linux:    export GITHUB_TOKEN="ghp_yourtoken"
  Get a free token at: https://github.com/settings/tokens
─────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import logging

import requests

logger = logging.getLogger(__name__)

GITHUB_API_BASE = "https://api.github.com"

# Map primary language → binary feature fields used by the model
LANG_TO_FEATURE = {
    "python":     "lang_python",
    "java":       "lang_java",
    "javascript": "lang_javascript",
    "typescript": "lang_javascript",
    "c++":        "lang_c_cpp",
    "c":          "lang_c_cpp",
    "kotlin":     "mobile_kotlin",
    "dart":       "mobile_flutter",
}

# Map topic/description keywords → skill score field
TOPIC_TO_SKILL = {
    "machine-learning":  "data_science_ml_skill",
    "deep-learning":     "data_science_ml_skill",
    "tensorflow":        "data_science_ml_skill",
    "pytorch":           "data_science_ml_skill",
    "data-science":      "data_analytics_skill",
    "data-analysis":     "data_analytics_skill",
    "web-development":   "web_dev_skill",
    "api":               "api_design_skill",
    "microservices":     "system_design_score",
    "cloud":             "cloud_devops_skill",
    "devops":            "cloud_devops_skill",
    "cybersecurity":     "cybersecurity_skill",
    "security":          "cybersecurity_skill",
    "mobile":            "mobile_dev_skill",
    "android":           "mobile_dev_skill",
    "embedded":          "embedded_c_cpp_skill",
    "firmware":          "embedded_c_cpp_skill",
    "database":          "db_sql_skill",
    "ui":                "ui_ux_design_skill",
    "testing":           "qa_testing_skill",
    "automation":        "qa_testing_skill",
}

# Tool keyword → binary flag feature
TOOL_TO_FEATURE = {
    "react":       "frontend_react",
    "angular":     "frontend_angular",
    "nodejs":      "backend_node",
    "node":        "backend_node",
    "django":      "backend_django",
    "spring":      "backend_spring",
    "postgresql":  "db_postgres",
    "spark":       "data_tool_spark",
    "airflow":     "data_tool_airflow",
    "kafka":       "data_tool_kafka",
    "aws":         "cloud_aws",
    "azure":       "cloud_azure",
    "docker":      "devops_docker",
    "kubernetes":  "devops_kubernetes",
    "terraform":   "devops_terraform",
    "prometheus":  "observability_prometheus",
    "grafana":     "observability_grafana",
    "selenium":    "testing_tool_selenium",
    "jmeter":      "testing_tool_jmeter",
    "wireshark":   "security_tool_wireshark",
    "burpsuite":   "security_tool_burpsuite",
    "flutter":     "mobile_flutter",
}


def _build_headers() -> dict:
    headers = {
        "Accept":     "application/vnd.github.v3+json",
        "User-Agent": "CareerAI-v2",
    }
    tok = os.environ.get("GITHUB_TOKEN", "").strip()
    if tok:
        headers["Authorization"] = f"token {tok}"
        logger.debug("GitHub: using authenticated token")
    else:
        logger.debug("GitHub: no token — anonymous (60 req/hr limit)")
    return headers


def fetch_github_profile(username: str) -> tuple[dict | None, str | None]:
    """
    Fetch GitHub user profile and repos, extract ML-relevant features.

    Returns:
        (result_dict, None)           on success
        (None,        error_message)  on any failure

    result_dict keys:
        extracted  — dict of feature_name → value (ready for predict_top3)
        summary    — dict of display info (avatar, bio, top_languages, etc.)
    """
    # Sanitise input
    username = username.strip().lstrip("@").split("/")[-1]
    if not username or len(username) > 39:
        return None, "Please enter a valid GitHub username (max 39 characters)."

    headers = _build_headers()

    # ── Fetch user profile ────────────────────────────────────────────────────
    try:
        user_resp = requests.get(
            f"{GITHUB_API_BASE}/users/{username}",
            headers=headers,
            timeout=10,
        )
    except requests.exceptions.ConnectionError:
        return None, "Cannot reach GitHub API — check your internet connection."
    except requests.exceptions.Timeout:
        return None, "GitHub API timed out — try again in a moment."
    except Exception as exc:
        logger.exception("Unexpected GitHub error")
        return None, f"GitHub error: {str(exc)[:120]}"

    # Status-code handling — each case has a distinct, actionable message
    if user_resp.status_code == 404:
        return None, f"GitHub user '{username}' not found — check the spelling."
    if user_resp.status_code == 403:
        remaining = user_resp.headers.get("X-RateLimit-Remaining", "?")
        reset_ts  = user_resp.headers.get("X-RateLimit-Reset", "")
        msg = (
            f"GitHub rate limit hit (remaining: {remaining}). "
            "Anonymous limit is 60 requests/hour. "
            "Fix: set GITHUB_TOKEN environment variable with a token from "
            "https://github.com/settings/tokens (free, no special scopes needed)."
        )
        if reset_ts.isdigit():
            import datetime
            reset_dt = datetime.datetime.fromtimestamp(int(reset_ts))
            msg += f"  Resets at {reset_dt.strftime('%H:%M:%S')}."
        return None, msg
    if user_resp.status_code == 401:
        return None, "GitHub authentication failed — check your GITHUB_TOKEN value."
    if user_resp.status_code != 200:
        return None, f"GitHub API error {user_resp.status_code} — try again."

    user = user_resp.json()
    logger.info("GitHub: fetched profile for @%s (%d repos)",
                user.get("login"), user.get("public_repos", 0))

    # ── Fetch repos ───────────────────────────────────────────────────────────
    try:
        repos_resp = requests.get(
            f"{GITHUB_API_BASE}/users/{username}/repos",
            headers=headers,
            params={"per_page": 100, "sort": "pushed", "type": "owner"},
            timeout=10,
        )
        repos = repos_resp.json() if repos_resp.status_code == 200 else []
        # Filter out forks — they don't represent the user's own work
        repos = [r for r in repos if isinstance(r, dict) and not r.get("fork")]
    except Exception:
        repos = []
        logger.warning("GitHub: could not fetch repos for @%s", username)

    # ── Feature extraction ────────────────────────────────────────────────────
    extracted: dict = {}

    # Basic activity features
    public_repos = int(user.get("public_repos", 0))
    extracted["github_repos_count"] = public_repos
    extracted["project_count"]      = min(public_repos, 20)

    # Language frequency counter
    lang_counts:  dict[str, int] = {}
    tool_flags:   set = set()
    topic_scores: dict[str, int] = {}

    # Project type buckets
    project_types: dict[str, int] = {
        "projects_ml_ai":          0,
        "projects_data_analytics": 0,
        "projects_backend":        0,
        "projects_frontend":       0,
        "projects_fullstack":      0,
        "projects_mobile_android": 0,
        "projects_security_defense": 0,
        "projects_cloud":          0,
        "projects_devops":         0,
        "projects_embedded":       0,
    }

    total_stars = 0

    for repo in repos:
        # Stars
        total_stars += int(repo.get("stargazers_count", 0))

        # Language
        lang = (repo.get("language") or "").lower()
        if lang:
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

            # Set binary language flag
            for kw, feat in LANG_TO_FEATURE.items():
                if kw == lang:
                    tool_flags.add(feat)

        # Topics + description: extract tool flags and skill hints
        topics      = repo.get("topics", []) or []
        description = (repo.get("description") or "").lower()
        name_lower  = repo.get("name", "").lower()
        text        = " ".join(topics) + " " + description + " " + name_lower

        for kw, feat in TOOL_TO_FEATURE.items():
            if kw in text or kw in lang:
                tool_flags.add(feat)

        for kw, skill in TOPIC_TO_SKILL.items():
            if kw in text:
                topic_scores[skill] = topic_scores.get(skill, 0) + 1

        # Classify project type from topic/description keywords
        if any(w in text for w in ["machine-learning", "deep-learning", "ml", "ai", "nlp",
                                     "pytorch", "tensorflow", "llm"]):
            project_types["projects_ml_ai"] += 1
        elif any(w in text for w in ["data", "analytics", "dashboard",
                                      "pandas", "etl", "airflow"]):
            project_types["projects_data_analytics"] += 1
        elif any(w in text for w in ["fullstack", "full-stack", "mern",
                                      "django", "react", "next"]):
            project_types["projects_fullstack"] += 1
        elif any(w in text for w in ["frontend", "css", "html", "vue",
                                      "angular", "tailwind"]):
            project_types["projects_frontend"] += 1
        elif any(w in text for w in ["backend", "api", "rest", "graphql",
                                      "server", "microservice", "node",
                                      "spring", "flask"]):
            project_types["projects_backend"] += 1
        elif any(w in text for w in ["android", "mobile", "ios",
                                      "flutter", "kotlin", "swift"]):
            project_types["projects_mobile_android"] += 1
        elif any(w in text for w in ["security", "ctf", "pentest",
                                      "exploit", "hack"]):
            project_types["projects_security_defense"] += 1
        elif any(w in text for w in ["cloud", "aws", "azure", "gcp",
                                      "terraform", "kubernetes"]):
            project_types["projects_devops"] += 1
        elif any(w in text for w in ["embedded", "firmware", "arduino",
                                      "stm32", "rtos"]):
            project_types["projects_embedded"] += 1

    # Write binary tool flags
    for feat in tool_flags:
        extracted[feat] = 1

    # Write language binary flags from lang_counts
    for lang, feat in LANG_TO_FEATURE.items():
        if lang in lang_counts:
            extracted[feat] = 1

    # Skill scores from topic analysis (capped at 8.5)
    for skill, count in topic_scores.items():
        extracted[skill] = max(extracted.get(skill, 0), min(2.5 + count * 1.2, 8.5))

    # Top-language inferred skill scores (conservative)
    top_langs = sorted(lang_counts.items(), key=lambda x: -x[1])
    for lang, _ in top_langs[:3]:
        if lang == "python":
            extracted.setdefault("programming_skill", 7.0)
            extracted.setdefault("data_analytics_skill", 5.0)
        elif lang in ("javascript", "typescript"):
            extracted.setdefault("programming_skill", 6.5)
            extracted.setdefault("web_dev_skill", 7.0)
        elif lang in ("java", "kotlin"):
            extracted.setdefault("programming_skill", 6.5)
        elif lang in ("c", "c++"):
            extracted.setdefault("programming_skill", 6.0)
            extracted.setdefault("embedded_c_cpp_skill", 5.0)

    # Write non-zero project types
    for k, v in project_types.items():
        if v > 0:
            extracted[k] = v

    # GitHub activity proxy (stars + repo count heuristic for commits)
    extracted["github_commits_90d"] = (
        min(150 + total_stars * 2, 400) if total_stars > 50
        else max(50, len(repos) * 8)
    )

    # Certification from bio
    bio = (user.get("bio") or "").lower()
    cert_count = len(re.findall(r'certif|certified|comptia|cisco|oscp', bio))
    if cert_count:
        extracted["certifications_total"] = min(cert_count * 2, 6)

    # ── Summary (for frontend card display) ──────────────────────────────────
    summary = {
        "username":     user.get("login"),
        "name":         user.get("name") or user.get("login"),
        "avatar_url":   user.get("avatar_url", ""),
        "bio":          user.get("bio") or "",
        "public_repos": public_repos,
        "followers":    user.get("followers", 0),
        "following":    user.get("following", 0),
        "location":     user.get("location") or "",
        "total_stars":  total_stars,
        "non_fork_repos": len(repos),
        "top_languages":  [l for l, _ in top_langs[:5]],
        "project_types":  {k: v for k, v in project_types.items() if v > 0},
        "detected_tools": sorted(tool_flags),
    }

    logger.info(
        "GitHub: extracted %d features, top_langs=%s, stars=%d",
        len(extracted), [l for l, _ in top_langs[:3]], total_stars,
    )

    return {"extracted": extracted, "summary": summary}, None
