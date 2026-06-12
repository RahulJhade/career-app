"""
utils/validators.py
─────────────────────────────────────────────────────────────────────────────
WHY THIS EXISTS:
  Invalid inputs cause silent failures — the model receives garbage and
  returns a confident-sounding but wrong prediction.

  Centralising validation here means:
    1. app.py stays clean (no inline if/else guards)
    2. Validation logic is testable in isolation
    3. Error messages are user-facing, not stack traces

FAILURE PREVENTION:
  • All-zero input   → detected and flagged (low-signal warning to user)
  • Out-of-range     → clamped to [min, max] before reaching the model
  • Wrong types      → coerced to float; non-numeric strings → 0
─────────────────────────────────────────────────────────────────────────────
"""

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

# ─── Field definitions ────────────────────────────────────────────────────────
# (field_name, min_val, max_val)
SKILL_RANGES: dict[str, tuple[float, float]] = {
    # Skills: 0–10
    "programming_skill":       (0, 10),
    "web_dev_skill":           (0, 10),
    "mobile_dev_skill":        (0, 10),
    "data_analytics_skill":    (0, 10),
    "data_science_ml_skill":   (0, 10),
    "db_sql_skill":            (0, 10),
    "cloud_devops_skill":      (0, 10),
    "cybersecurity_skill":     (0, 10),
    "networking_sysadmin_skill": (0, 10),
    "ui_ux_design_skill":      (0, 10),
    "qa_testing_skill":        (0, 10),
    "business_analysis_skill": (0, 10),
    "system_design_score":     (0, 10),
    "distributed_systems_knowledge_score": (0, 10),
    "cloud_arch_patterns_score": (0, 10),
    "data_modeling_skill":     (0, 10),
    "api_design_skill":        (0, 10),
    "embedded_c_cpp_skill":    (0, 10),
    "siem_experience_score":   (0, 10),
    "rtos_experience_score":   (0, 10),
    "firmware_debugging_skill": (0, 10),
    # Interests: 0–10
    "interest_dev_overall":             (0, 10),
    "interest_data_overall":            (0, 10),
    "interest_cloud_infra_overall":     (0, 10),
    "interest_cybersecurity":           (0, 10),
    "interest_ui_ux_design":            (0, 10),
    "interest_business_and_management": (0, 10),
    # Soft skills: 0–10
    "teamwork_behavior":      (0, 10),
    "learning_motivation":    (0, 10),
    "communication_skill":    (0, 10),
    # Academic
    "math_scores":             (0, 100),
    "cs_fundamentals_scores":  (0, 100),
    "cognitive_ability_score": (0, 10),
    "project_complexity":      (0, 10),
    "cgpa":                    (0, 10),
    # Counts
    "project_count":            (0, 50),
    "github_commits_90d":       (0, 1000),
    "github_repos_count":       (0, 500),
    "certifications_total":     (0, 20),
    "internship_experience_count": (0, 10),
    "incident_response_cases":  (0, 100),
    "vuln_assessments_done":    (0, 100),
    "pentest_tools_known_count": (0, 30),
    "compliance_frameworks_known_count": (0, 20),
    "microcontroller_projects_count": (0, 50),
    # 0–1 scale
    "professional_discipline_score": (0, 1),
    # Binary flags (0 or 1 — clamped)
    "lang_python":     (0, 1), "lang_java":       (0, 1),
    "lang_javascript": (0, 1), "lang_c_cpp":      (0, 1),
    "lang_sql":        (0, 1),
    "frontend_react":  (0, 1), "frontend_angular": (0, 1),
    "backend_node":    (0, 1), "backend_django":   (0, 1),
    "backend_spring":  (0, 1), "db_postgres":      (0, 1),
    "data_tool_spark": (0, 1), "data_tool_airflow": (0, 1),
    "data_tool_kafka": (0, 1),
    "cloud_aws":       (0, 1), "cloud_azure":      (0, 1),
    "devops_docker":   (0, 1), "devops_kubernetes": (0, 1),
    "devops_terraform": (0, 1),
    "observability_prometheus": (0, 1), "observability_grafana": (0, 1),
    "security_tool_siem": (0, 1), "security_tool_wireshark": (0, 1),
    "security_tool_burpsuite": (0, 1),
    "testing_tool_selenium": (0, 1), "testing_tool_jmeter": (0, 1),
    "mobile_kotlin": (0, 1), "mobile_flutter": (0, 1),
    # Project counts by type
    "projects_backend":  (0, 30), "projects_frontend": (0, 30),
    "projects_fullstack": (0, 30), "projects_mobile_android": (0, 30),
    "projects_data_analytics": (0, 30), "projects_data_engineering": (0, 30),
    "projects_ml_ai": (0, 30), "projects_cloud": (0, 30),
    "projects_devops": (0, 30), "projects_security_defense": (0, 30),
    "projects_security_offense": (0, 30), "projects_embedded": (0, 30),
}

# Skill fields whose sum is checked for the all-zero signal detection
SIGNAL_FIELDS = [
    "programming_skill", "data_analytics_skill", "data_science_ml_skill",
    "cloud_devops_skill", "cybersecurity_skill", "ui_ux_design_skill",
    "web_dev_skill", "mobile_dev_skill", "system_design_score",
    "interest_dev_overall", "interest_data_overall", "interest_cybersecurity",
    "project_count", "github_commits_90d", "cgpa",
]


@dataclass
class ValidationResult:
    cleaned:  dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors:   list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.errors) == 0


def validate_and_clean(raw: dict) -> ValidationResult:
    """
    Validate, coerce, and clamp every numeric field.

    Returns:
      ValidationResult with:
        .cleaned  — safe dict ready for the model
        .warnings — non-fatal issues shown to user (e.g. low-signal)
        .errors   — fatal issues (empty if all OK)

    FAILURE PREVENTION:
      • NaN / None values → treated as 0
      • Strings like "7.5" → coerced to float
      • Negative values   → clamped to 0
      • Out-of-range      → clamped to max
    """
    result = ValidationResult()
    cleaned: dict = {}

    for key, value in raw.items():
        if key == "student_name":
            cleaned[key] = str(value)[:80]
            continue

        # Coerce to float
        try:
            fval = float(value)
        except (TypeError, ValueError):
            fval = 0.0
            logger.debug("validate: non-numeric value for '%s' → 0", key)

        # NaN guard
        import math
        if math.isnan(fval) or math.isinf(fval):
            fval = 0.0

        # Clamp to [min, max] using field definition, or default 0–1000
        lo, hi = SKILL_RANGES.get(key, (0.0, 1000.0))
        clamped = max(lo, min(fval, hi))
        if clamped != fval:
            logger.debug("validate: clamped %s: %.2f → %.2f", key, fval, clamped)

        cleaned[key] = clamped

    result.cleaned = cleaned

    # ── All-zero signal detection ─────────────────────────────────────────────
    signal_sum = sum(float(cleaned.get(f, 0)) for f in SIGNAL_FIELDS)
    if signal_sum < 0.5:
        result.warnings.append(
            "Your profile has very few signals — prediction may be imprecise. "
            "Upload a resume, connect GitHub, or fill in your skills manually."
        )
        logger.warning("validate: all-signal-zero profile submitted")

    return result
