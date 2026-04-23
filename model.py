"""
model.py — Model training, governed prediction, and SHAP explanations.

Encapsulates:
  - Synthetic data generation (loan‑approval dataset with intentional bias).
  - Logistic Regression training.
  - Governed prediction with confidence & risk classification.
  - SHAP‑based explainability for individual predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FEATURE_COLS = ["income", "credit_score", "age", "debt_ratio", "years_emp"]
CONFIDENCE_THRESHOLD = 0.6

# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def generate_data(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Produce a synthetic loan‑approval dataset with protected attributes
    and intentional demographic bias (mirrors the notebook's
    ``generate_fairness_benchmark``).
    """
    rng = np.random.RandomState(seed)

    gender = rng.choice(["Male", "Female", "Non-binary"], n, p=[0.50, 0.45, 0.05])
    credit_score = rng.normal(680, 80, n).clip(300, 850).astype(int)
    income = rng.lognormal(10.8, 0.6, n).astype(int)
    age = rng.randint(18, 70, n)
    debt_ratio = rng.beta(2, 5, n).round(3)
    years_emp = rng.poisson(7, n)

    # Ground truth based on objective features
    score = (
        (credit_score - 300) / 550 * 0.5
        + (income / 200_000) * 0.3
        - debt_ratio * 0.2
    )
    y_true = (score + rng.normal(0, 0.08, n) > 0.38).astype(int)

    # Inject demographic bias into predictions
    bias_term = np.zeros(n)
    bias_term[gender == "Female"] -= 0.07
    bias_term[gender == "Non-binary"] -= 0.12

    pred_prob = (score + bias_term + rng.normal(0, 0.06, n)).clip(0, 1)
    y_pred = (pred_prob > 0.40).astype(int)

    return pd.DataFrame({
        "gender": gender,
        "credit_score": credit_score,
        "income": income,
        "age": age,
        "debt_ratio": debt_ratio,
        "years_emp": years_emp,
        "y_true": y_true,
        "y_pred": y_pred,
        "y_pred_prob": pred_prob.round(4),
    })


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(df: pd.DataFrame):
    """
    Train a simple Logistic Regression on the objective features.

    Returns
    -------
    model : LogisticRegression
    scaler : StandardScaler
    """
    X = df[FEATURE_COLS].values
    y = df["y_true"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)

    return model, scaler


# ---------------------------------------------------------------------------
# Risk classification helper
# ---------------------------------------------------------------------------

def _classify_risk(probability: float) -> str:
    if probability > 0.8:
        return "Low Risk"
    elif probability >= 0.5:
        return "Medium Risk"
    else:
        return "High Risk"


# ---------------------------------------------------------------------------
# Governed prediction
# ---------------------------------------------------------------------------

def governed_predict(model, scaler, input_dict: dict) -> dict:
    """
    Perform a *governed* prediction.

    Parameters
    ----------
    model : fitted LogisticRegression
    scaler : fitted StandardScaler
    input_dict : dict with keys matching ``FEATURE_COLS``

    Returns
    -------
    dict with:
        prediction   – 0 or 1
        probability  – float  (probability of positive class)
        confidence   – str    ("High" or "Low")
        decision     – str    ("Approved" / "Denied" / "Review Required")
        risk_level   – str    ("High Risk" / "Medium Risk" / "Low Risk")
    """
    features = np.array([[input_dict[c] for c in FEATURE_COLS]])
    features_scaled = scaler.transform(features)

    proba = model.predict_proba(features_scaled)[0]  # [P(0), P(1)]
    prob_positive = float(proba[1])
    prediction = int(prob_positive >= 0.5)

    # Confidence gate
    confidence = max(proba)
    if confidence < CONFIDENCE_THRESHOLD:
        decision = "Review Required"
        confidence_label = "Low"
    else:
        decision = "Approved" if prediction == 1 else "Denied"
        confidence_label = "High"

    risk_level = _classify_risk(prob_positive)

    return {
        "prediction": prediction,
        "probability": round(prob_positive, 4),
        "confidence": confidence_label,
        "decision": decision,
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# SHAP Explainability
# ---------------------------------------------------------------------------

def compute_shap_explanation(model, scaler, input_dict: dict) -> dict:
    """
    Compute SHAP values for a single prediction to explain *why* the
    model approved or denied the loan.

    Uses ``shap.LinearExplainer`` which is the most efficient explainer
    for linear models like Logistic Regression.

    Parameters
    ----------
    model : fitted LogisticRegression
    scaler : fitted StandardScaler
    input_dict : dict with keys matching ``FEATURE_COLS``

    Returns
    -------
    dict with:
        feature_names  – list[str]
        shap_values    – list[float]   (per‑feature contributions)
        base_value     – float         (model's average prediction)
        raw_features   – list[float]   (unscaled input values)
        explanation    – str           (human‑readable textual summary)
    """
    import shap  # lazy import to keep startup fast

    # Prepare input in the same format used for prediction
    raw_values = [input_dict[c] for c in FEATURE_COLS]
    features = np.array([raw_values])
    features_scaled = scaler.transform(features)

    # Build background dataset from scaler statistics (mean ± 0)
    # This avoids needing the full training set at runtime
    background = scaler.mean_.reshape(1, -1)  # shape (1, n_features)

    # LinearExplainer is exact for linear models & very fast
    explainer = shap.LinearExplainer(model, background)
    shap_vals = explainer.shap_values(features_scaled)

    # For binary classification, shap_values returns a list of two arrays
    # (one per class). We want the positive‑class (index 1) explanation.
    if isinstance(shap_vals, list):
        sv = shap_vals[1][0]   # shape (n_features,)
    else:
        sv = shap_vals[0]      # single array fallback

    base = float(explainer.expected_value[1]) if hasattr(
        explainer.expected_value, '__len__'
    ) else float(explainer.expected_value)

    # ------- Build human‑readable textual explanation -------
    # Pretty feature labels for display
    _LABELS = {
        "income": "Income",
        "credit_score": "Credit Score",
        "age": "Age",
        "debt_ratio": "Debt Ratio",
        "years_emp": "Years Employed",
    }

    # Sort features by absolute SHAP impact (descending)
    indexed = sorted(
        enumerate(sv), key=lambda x: abs(x[1]), reverse=True
    )

    positives = []
    negatives = []
    for idx, val in indexed:
        label = _LABELS.get(FEATURE_COLS[idx], FEATURE_COLS[idx])
        if val > 0:
            positives.append(f"{label} (value={raw_values[idx]})")
        elif val < 0:
            negatives.append(f"{label} (value={raw_values[idx]})")

    parts = []
    if positives:
        parts.append(
            "**Positive contributors:** " + ", ".join(positives)
            + " pushed the prediction toward approval."
        )
    if negatives:
        parts.append(
            "**Negative contributors:** " + ", ".join(negatives)
            + " reduced approval chances."
        )
    explanation = " ".join(parts) if parts else "No dominant feature found."

    return {
        "feature_names": [_LABELS.get(c, c) for c in FEATURE_COLS],
        "shap_values": sv.tolist(),
        "base_value": base,
        "raw_features": raw_values,
        "explanation": explanation,
    }
