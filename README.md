# Responsible AI Governance Framework for Loan Approval

## Overview

This project is an **end-to-end Responsible AI system** for loan approval that goes beyond traditional machine learning by incorporating **fairness, governance, explainability, and monitoring**.

Instead of blindly trusting model predictions, this system ensures that every decision is:

* Fair
* Explainable
* Auditable
* Secure

---

## Key Features

### 1. Machine Learning Model

* Logistic Regression for loan approval prediction
* Outputs:

  * Approval / Denial
  * Probability
  * Risk Level
  * Confidence

---

### 2. Fairness & Bias Detection

* Uses:

  * Demographic Parity
  * Equalized Odds
* Computes **dynamic fairness per prediction**
* Detects bias based on gender

---

### 3. Policy Engine (Core Governance)

* Rule-based decision system
* Enforces:

  * Access control
  * Bias thresholds
  * Confidence checks

#### Decision Levels:

* ALLOW → Safe decision
* REVIEW → Needs human intervention
* REJECT → Violates policy

---

### 4. Explainability (SHAP)

* Provides feature-level explanations
* Shows:

  * Which features increased approval
  * Which features decreased approval

---

### 5. Logging & Audit System

* Tracks:

  * User actions
  * Model predictions
  * Bias scores
  * Policy violations
* Enables full auditability

---

### 6. Anomaly Detection

* Uses Isolation Forest
* Detects unusual or suspicious decisions

---

###  7. Compliance Dashboard

* Displays:

  * Total requests
  * Average bias score
  * Violations
  * Anomalies

---

## System Architecture

```
User Input
   ↓
Access Control
   ↓
ML Model (Prediction)
   ↓
Fairness Check
   ↓
Policy Engine
   ↓
Explainability (SHAP)
   ↓
Logging
   ↓
Dashboard & Monitoring
```

---

## Project Structure

```
├── app.py          # Streamlit UI & pipeline orchestration
├── model.py        # Data generation, training, prediction, SHAP
├── governance.py   # Fairness, policy engine, RBAC, anomaly detection
├── logs.py         # Logging and compliance reporting
```

---

## Dataset

* **Synthetic dataset (~2000 samples)**
* Features:

  * Income
  * Credit Score
  * Age
  * Debt Ratio
  * Years Employed
  * Gender (sensitive attribute)

### Bias Injection

* Intentional bias added:

  * Female: slight penalty
  * Non-binary: higher penalty
* Used to simulate real-world unfairness

---

## Core Concepts Used

* Responsible AI
* Fairness Metrics
* Logistic Regression
* SHAP Explainability
* Role-Based Access Control (RBAC)
* Policy-Based Governance
* Isolation Forest (Anomaly Detection)

---

## How Decision is Made

1. Model predicts approval probability
2. Risk & confidence are computed
3. Fairness module checks bias
4. Policy engine evaluates rules
5. Final decision is given:

   * Approved
   * Review Required
   * Rejected

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Why This Project?

Traditional ML systems:

* Ignore bias
* Lack transparency
* No governance

This system:

* Detects bias
* Explains decisions
* Enforces policies
* Logs everything

---

## Key Insight

> This project transforms a simple ML model into a **Responsible AI system** by adding governance, fairness, and accountability layers.

---

## Limitations

* Uses synthetic data
* Simple ML model
* Rule-based governance (not adaptive)

---

## Future Improvements

* Use real-world datasets
* Add deep learning models
* Implement adaptive/learning policy engine
* Deploy on cloud (AWS/Azure)

---

## Author

Aadish Bane

B.Tech Electronics & Communication Engineering
SVNIT Surat

---

