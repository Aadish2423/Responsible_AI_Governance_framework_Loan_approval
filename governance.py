"""
governance.py — Policy engine, fairness metrics, RBAC, and anomaly detection.

Centralises all governance logic for the RAI framework.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from fairlearn.metrics import (
    demographic_parity_difference,
    equalized_odds_difference,
)

# ===================================================================
# Access Control (RBAC)
# ===================================================================

# Allowed actions per role
_ROLE_PERMISSIONS: dict[str, list[str]] = {
    "admin": ["predict", "view_logs", "view_fairness", "view_compliance", "full"],
    "auditor": ["view_logs", "view_fairness", "view_compliance"],
    "user": ["predict"],
}


class AccessControl:
    """Simple role‑based access control."""

    @staticmethod
    def check_access(role: str, action: str) -> bool:
        perms = _ROLE_PERMISSIONS.get(role, [])
        return action in perms or "full" in perms

    @staticmethod
    def get_roles() -> list[str]:
        return list(_ROLE_PERMISSIONS.keys())


# ===================================================================
# Fairness metrics
# ===================================================================

# Tiered bias thresholds for balanced governance
BIAS_THRESHOLD = 0.1          # informational warning
BIAS_REVIEW_THRESHOLD = 0.18  # triggers "Review Required"
BIAS_REJECT_THRESHOLD = 0.30  # triggers hard rejection


def compute_fairness(
    df: pd.DataFrame,
    sensitive_attr: str = "gender",
    y_true_col: str = "y_true",
    y_pred_col: str = "y_pred",
) -> dict:
    """
    Compute fairness metrics on a given dataset (typically the full
    training/benchmark data).

    Returns
    -------
    dict with:
        demographic_parity_diff – float
        equalized_odds_diff     – float
        bias_score              – float  (max of the two)
        bias_detected           – bool
    """
    y_true = df[y_true_col]
    y_pred = df[y_pred_col]
    sensitive = df[sensitive_attr]

    dp_diff = abs(demographic_parity_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
    ))
    eo_diff = abs(equalized_odds_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
    ))

    bias_score = round(max(dp_diff, eo_diff), 4)

    return {
        "demographic_parity_diff": round(dp_diff, 4),
        "equalized_odds_diff": round(eo_diff, 4),
        "bias_score": bias_score,
        "bias_detected": bias_score > BIAS_THRESHOLD,
    }


def compute_dynamic_fairness(
    model,
    scaler,
    df: pd.DataFrame,
    current_input: dict,
    current_gender: str,
    feature_cols: list[str],
    sensitive_attr: str = "gender",
    sample_size: int = 200,
    seed: int = 42,
) -> dict:
    """
    Compute a *dynamic* bias score that reflects the current prediction
    in context — not just the static training data.

    How it works:
      1. Sample a small subset from the training data.
      2. Append the current applicant's record to that sample.
      3. Generate *model* predictions for all rows (including the new one).
      4. Compute fairness metrics on this combined set.

    This gives a bias score that varies based on the applicant's profile
    and the model's behaviour on a representative population — much more
    realistic than the always-constant training-data score.

    Returns
    -------
    dict with same keys as compute_fairness, plus:
        bias_level – str ("Low" / "Moderate" / "Severe")
    """
    rng = np.random.RandomState(seed)

    # 1) Sample from training data
    sample = df.sample(n=min(sample_size, len(df)), random_state=rng)

    # 2) Build the current applicant's row and append
    current_row = {col: current_input[col] for col in feature_cols}
    current_row[sensitive_attr] = current_gender
    current_row["y_true"] = 1  # assume ground truth positive for fairness calc
    combined = pd.concat([sample, pd.DataFrame([current_row])], ignore_index=True)

    # 3) Generate model predictions for the combined set
    X = combined[feature_cols].values
    X_scaled = scaler.transform(X)
    combined["y_pred"] = model.predict(X_scaled)

    # 4) Compute fairness on combined set
    sensitive = combined[sensitive_attr]
    y_true = combined["y_true"]
    y_pred = combined["y_pred"]

    dp_diff = abs(demographic_parity_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
    ))
    eo_diff = abs(equalized_odds_difference(
        y_true=y_true, y_pred=y_pred, sensitive_features=sensitive
    ))

    bias_score = round(max(dp_diff, eo_diff), 4)

    # Classify severity
    if bias_score > BIAS_REJECT_THRESHOLD:
        bias_level = "Severe"
    elif bias_score >= BIAS_REVIEW_THRESHOLD and bias_score < BIAS_REJECT_THRESHOLD:
        bias_level = "Moderate"
    elif bias_score > BIAS_THRESHOLD:
        bias_level = "Low"
    else:
        bias_level = "None"

    return {
        "demographic_parity_diff": round(dp_diff, 4),
        "equalized_odds_diff": round(eo_diff, 4),
        "bias_score": bias_score,
        "bias_detected": bias_score > BIAS_THRESHOLD,
        "bias_level": bias_level,
    }


# ===================================================================
# Policy Engine — Tiered decision logic
# ===================================================================

class PolicyEngine:
    """
    Enforce governance policies on each prediction request.

    Uses a **tiered** approach instead of binary reject:
      - ALLOW       → no significant issues
      - REVIEW      → moderate bias or low model confidence
      - REJECT      → severe bias or unauthorized access
    """

    @staticmethod
    def evaluate(
        prediction_result: dict,
        bias_score: float,
        bias_level: str,
        user_role: str,
        action: str = "predict",
    ) -> dict:
        """
        Evaluate governance policies and produce a final decision.

        Parameters
        ----------
        prediction_result : dict from governed_predict()
        bias_score        : float  (dynamic bias score)
        bias_level        : str    ("None" / "Low" / "Moderate" / "Severe")
        user_role         : str
        action            : str

        Returns
        -------
        dict with:
            final_decision – str ("Approved" / "Denied" /
                                  "Review Required" / "Rejected — Bias" /
                                  "Policy Violation")
            violations     – list[str]
            governance_status – str ("ALLOW" / "REVIEW" / "REJECT")
        """
        violations: list[str] = []
        governance_status = "ALLOW"

        # --- 1) RBAC check (hard block) ---
        if not AccessControl.check_access(user_role, action):
            violations.append(
                f"Unauthorized: role '{user_role}' cannot perform '{action}'"
            )
            governance_status = "REJECT"

        # --- 2) Bias check (tiered) ---
        if bias_level == "Severe":
            violations.append(
                f"Severe bias detected: score={bias_score} > {BIAS_REJECT_THRESHOLD} — automatic rejection"
            )
            governance_status = "REJECT"
        elif bias_level == "Moderate":
            violations.append(
                f"Moderate bias detected: score={bias_score} > {BIAS_REVIEW_THRESHOLD} — manual review recommended"
            )
            if governance_status != "REJECT":
                governance_status = "REVIEW"
        elif bias_level == "Low":
            violations.append(
                f"Low bias detected: score={bias_score} > {BIAS_THRESHOLD} — within acceptable range, proceed with caution"
            )
            # Low bias is informational — does NOT change governance_status

        # --- 3) Confidence check ---
        if prediction_result.get("confidence") == "Low":
            violations.append("Low model confidence — manual review required")
            if governance_status != "REJECT":
                governance_status = "REVIEW"

        # --- 4) Final decision ---
        if governance_status == "REJECT":
            if any("Unauthorized" in v for v in violations):
                final_decision = "Policy Violation"
            else:
                final_decision = "Rejected — Bias"
        elif governance_status == "REVIEW":
            final_decision = "Review Required"
        else:
            # ALLOW — pass through the model's decision
            final_decision = prediction_result.get("decision", "Approved")

        return {
            "final_decision": final_decision,
            "violations": violations,
            "governance_status": governance_status,
        }


# ===================================================================
# Anomaly Detection
# ===================================================================

def detect_anomalies(audit_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run Isolation Forest on audit logs to flag anomalous predictions.

    Expects columns: ``probability``, ``bias_score``.
    Adds an ``anomaly`` column (1 = anomaly, 0 = normal).
    """
    if audit_df.empty:
        return audit_df.assign(anomaly=pd.Series(dtype=int))

    numeric_cols = ["probability", "bias_score"]
    available = [c for c in numeric_cols if c in audit_df.columns]
    if not available:
        return audit_df.assign(anomaly=0)

    X = audit_df[available].values
    iso = IsolationForest(contamination=0.1, random_state=42)
    preds = iso.fit_predict(X)  # 1 = normal, -1 = anomaly
    audit_df = audit_df.copy()
    audit_df["anomaly"] = (preds == -1).astype(int)
    return audit_df
