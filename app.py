"""
app.py — Streamlit UI for the Responsible AI Governance Framework.

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

from model import generate_data, train_model, governed_predict, compute_shap_explanation, FEATURE_COLS
from governance import (
    AccessControl,
    compute_fairness,
    compute_dynamic_fairness,
    PolicyEngine,
    detect_anomalies,
    BIAS_THRESHOLD,
    BIAS_REVIEW_THRESHOLD,
    BIAS_REJECT_THRESHOLD,
)
from logs import Logger

# ===================================================================
# Page configuration
# ===================================================================
st.set_page_config(
    page_title="RAI Governance Framework",
    page_icon="🛡️",
    layout="wide",
)

# ===================================================================
# CSS — Dark premium dashboard theme
# ===================================================================
st.markdown("""
<style>
/* ---------- Google Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ---------- Global ---------- */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a1a2e 0%, #0f0f1a 100%) !important;
    border-right: 1px solid rgba(255,255,255,0.06);
}
section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    color: #e0e0ff !important;
}

/* ---------- Card mixin ---------- */
.card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    backdrop-filter: blur(12px);
}
.card-glow-green { box-shadow: 0 0 24px rgba(46,204,113,0.10); }
.card-glow-red   { box-shadow: 0 0 24px rgba(231,76,60,0.10); }
.card-glow-blue  { box-shadow: 0 0 24px rgba(52,152,219,0.10); }

/* ---------- Metric cards row ---------- */
.metric-row { display: flex; gap: 16px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 160px;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 20px 22px;
    text-align: center;
}
.metric-card .label {
    font-size: 12px; font-weight: 500; text-transform: uppercase;
    letter-spacing: 1.2px; color: #8892b0; margin-bottom: 8px;
}
.metric-card .value {
    font-size: 26px; font-weight: 700; color: #e0e0ff;
}
.metric-card .value.green  { color: #2ecc71; }
.metric-card .value.red    { color: #e74c3c; }
.metric-card .value.yellow { color: #f39c12; }
.metric-card .value.blue   { color: #3498db; }

/* ---------- Alert cards ---------- */
.alert-card {
    border-radius: 12px; padding: 16px 20px;
    margin-bottom: 10px; font-size: 14px; font-weight: 500;
    display: flex; align-items: center; gap: 10px;
}
.alert-red    { background: rgba(231,76,60,0.12); border-left: 4px solid #e74c3c; color: #f5b7b1; }
.alert-yellow { background: rgba(243,156,18,0.12); border-left: 4px solid #f39c12; color: #fdebd0; }
.alert-orange { background: rgba(230,126,34,0.12); border-left: 4px solid #e67e22; color: #fad7a0; }
.alert-blue   { background: rgba(52,152,219,0.12); border-left: 4px solid #3498db; color: #aed6f1; }
.alert-green  { background: rgba(46,204,113,0.12); border-left: 4px solid #2ecc71; color: #a9dfbf; }

/* ---------- Section headers ---------- */
.section-title {
    font-size: 20px; font-weight: 700; color: #e0e0ff;
    margin: 32px 0 6px 0; display: flex; align-items: center; gap: 10px;
}
.section-sub {
    font-size: 13px; color: #8892b0; margin-bottom: 18px;
}

/* ---------- Insight chips ---------- */
.insight-chip {
    display: flex; align-items: center; gap: 10px;
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
}
.insight-chip .feat { font-weight: 600; color: #e0e0ff; font-size: 14px; }
.insight-chip .impact { font-size: 13px; }
.impact-pos { color: #2ecc71; }
.impact-neg { color: #e74c3c; }

/* ---------- Feature overview ---------- */
.feat-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr)); gap: 12px; }
.feat-item {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 10px; padding: 14px 16px;
}
.feat-item .fl { font-size: 11px; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }
.feat-item .fv { font-size: 20px; font-weight: 700; color: #e0e0ff; margin-top: 4px; }

/* ---------- Hide default metric styling ---------- */
[data-testid="stMetric"] { background: rgba(255,255,255,0.04); border-radius: 12px; padding: 14px; border: 1px solid rgba(255,255,255,0.07); }

/* ---------- Header ---------- */
.hero { text-align: center; padding: 10px 0 4px 0; }
.hero h1 { font-size: 32px; font-weight: 700; color: #e0e0ff; margin-bottom: 4px; }
.hero p { color: #8892b0; font-size: 14px; letter-spacing: 2px; }

/* ---------- Divider ---------- */
.divider { border: none; border-top: 1px solid rgba(255,255,255,0.06); margin: 28px 0; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab"] { color: #8892b0 !important; }
.stTabs [aria-selected="true"] { color: #e0e0ff !important; }
</style>
""", unsafe_allow_html=True)


# ===================================================================
# Session-state initialisation (runs once per session)
# ===================================================================
if "initialized" not in st.session_state:
    df = generate_data()
    model, scaler = train_model(df)
    st.session_state["df"] = df
    st.session_state["model"] = model
    st.session_state["scaler"] = scaler
    st.session_state["logger"] = Logger()
    st.session_state["initialized"] = True

df: pd.DataFrame = st.session_state["df"]
model = st.session_state["model"]
scaler = st.session_state["scaler"]
logger: Logger = st.session_state["logger"]


# ===================================================================
# Helper — render HTML card
# ===================================================================
def _html(html_str: str):
    st.markdown(html_str, unsafe_allow_html=True)


# ===================================================================
# Sidebar — role + inputs
# ===================================================================
with st.sidebar:
    _html("""
    <div style="text-align:center;padding:16px 0 8px 0;">
        <span style="font-size:36px;">🛡️</span>
        <h2 style="margin:6px 0 2px 0;color:#e0e0ff;">RAI Governance</h2>
        <p style="color:#8892b0;font-size:12px;letter-spacing:1.5px;">CONTROL PANEL</p>
    </div>
    """)
    st.markdown("---")

    role = st.selectbox("🔐  Select Role", AccessControl.get_roles())

    _html('<p style="color:#8892b0;font-size:13px;font-weight:600;letter-spacing:1px;margin:20px 0 8px 0;">📋 INPUT FEATURES</p>')

    income = st.number_input("💰 Income ($)", min_value=0, value=60000, step=1000)
    credit_score = st.number_input("📊 Credit Score", min_value=300, max_value=850, value=700)
    age = st.number_input("🎂 Age", min_value=18, max_value=100, value=35)
    debt_ratio = st.slider("📉 Debt Ratio", 0.0, 1.0, 0.3, step=0.01)
    years_emp = st.number_input("💼 Years Employed", min_value=0, value=5)
    gender = st.selectbox("👤 Gender", ["Male", "Female", "Non-binary"])

    st.markdown("")
    predict_btn = st.button("🚀  Run Prediction", use_container_width=True, type="primary")


# ===================================================================
# Main — Hero header
# ===================================================================
_html("""
<div class="hero">
    <h1>🛡️ Responsible AI Governance Framework</h1>
    <p>FAIRNESS · ACCOUNTABILITY · SECURITY · COMPLIANCE</p>
</div>
<hr class="divider">
""")


# ===================================================================
# Prediction flow (LOGIC UNCHANGED)
# ===================================================================
if predict_btn:
    # 1) RBAC check
    if not AccessControl.check_access(role, "predict"):
        _html('<div class="alert-card alert-red">❌ <b>Unauthorized Access</b> — your role does not allow predictions.</div>')
        logger.add_user_log(user=role, action="predict", status="Unauthorized")
    else:
        # 2) Governed prediction
        input_data = {
            "income": income,
            "credit_score": credit_score,
            "age": age,
            "debt_ratio": debt_ratio,
            "years_emp": years_emp,
        }
        pred_result = governed_predict(model, scaler, input_data)

        # 3) Dynamic Fairness — evaluates bias in context of current prediction
        fairness = compute_dynamic_fairness(
            model=model,
            scaler=scaler,
            df=df,
            current_input=input_data,
            current_gender=gender,
            feature_cols=FEATURE_COLS,
            sensitive_attr="gender",
        )

        # 4) Tiered Policy Engine (ALLOW / REVIEW / REJECT)
        policy = PolicyEngine.evaluate(
            prediction_result=pred_result,
            bias_score=fairness["bias_score"],
            bias_level=fairness["bias_level"],
            user_role=role,
            action="predict",
        )

        # 5) Logging
        logger.add_audit_log(
            input_data=input_data,
            prediction=pred_result["prediction"],
            probability=pred_result["probability"],
            bias_score=fairness["bias_score"],
            decision=policy["final_decision"],
            violations=policy["violations"],
        )
        logger.add_user_log(user=role, action="predict", status=policy["final_decision"])

        # =============================================================
        # 6) FEATURE OVERVIEW CARD
        # =============================================================
        _html('<div class="section-title">📋 Applicant Profile</div>')
        _html(f"""
        <div class="feat-grid">
            <div class="feat-item"><div class="fl">💰 Income</div><div class="fv">${income:,.0f}</div></div>
            <div class="feat-item"><div class="fl">📊 Credit Score</div><div class="fv">{credit_score}</div></div>
            <div class="feat-item"><div class="fl">🎂 Age</div><div class="fv">{age}</div></div>
            <div class="feat-item"><div class="fl">📉 Debt Ratio</div><div class="fv">{debt_ratio:.0%}</div></div>
            <div class="feat-item"><div class="fl">💼 Years Employed</div><div class="fv">{years_emp}</div></div>
            <div class="feat-item"><div class="fl">👤 Gender</div><div class="fv">{gender}</div></div>
        </div>
        """)

        # =============================================================
        # 7) PREDICTION SUMMARY CARD
        # =============================================================
        is_approved = pred_result["prediction"] == 1
        pred_icon = "✅" if is_approved else "❌"
        pred_label = "Approved" if is_approved else "Denied"
        pred_color = "green" if is_approved else "red"

        risk = pred_result["risk_level"]
        risk_color = "green" if "Low" in risk else ("yellow" if "Medium" in risk else "red")

        conf = pred_result["confidence"]
        conf_color = "green" if conf == "High" else "yellow"

        bias = fairness["bias_score"]
        bias_level = fairness["bias_level"]
        bias_color = {"None": "green", "Low": "yellow", "Moderate": "yellow", "Severe": "red"}.get(bias_level, "yellow")

        gov_status = policy["governance_status"]
        gov_color = {"ALLOW": "green", "REVIEW": "yellow", "REJECT": "red"}.get(gov_status, "blue")

        # Map final decision to color
        fd = policy["final_decision"]
        fd_color = "green" if fd == "Approved" else ("red" if "Reject" in fd or "Denied" in fd or "Violation" in fd else "yellow")

        _html('<hr class="divider">')
        _html('<div class="section-title">📊 Prediction Summary</div>')
        _html(f"""
        <div class="metric-row">
            <div class="metric-card">
                <div class="label">Model Decision</div>
                <div class="value {pred_color}">{pred_icon} {pred_label}</div>
            </div>
            <div class="metric-card">
                <div class="label">Confidence</div>
                <div class="value {conf_color}">{conf}</div>
            </div>
            <div class="metric-card">
                <div class="label">Risk Level</div>
                <div class="value {risk_color}">{risk}</div>
            </div>
            <div class="metric-card">
                <div class="label">Bias Score</div>
                <div class="value {bias_color}">{bias} ({bias_level})</div>
            </div>
            <div class="metric-card">
                <div class="label">Governance</div>
                <div class="value {gov_color}">{gov_status}</div>
            </div>
            <div class="metric-card">
                <div class="label">Final Decision</div>
                <div class="value {fd_color}">{fd}</div>
            </div>
        </div>
        """)

        # =============================================================
        # 8) ALERTS — styled cards
        # =============================================================
        if policy["violations"]:
            _html('<hr class="divider">')
            _html('<div class="section-title">⚠️ Governance Alerts</div>')

            for v in policy["violations"]:
                if "Unauthorized" in v:
                    _html(f'<div class="alert-card alert-red">❌ {v}</div>')
                elif "Severe" in v:
                    _html(f'<div class="alert-card alert-red">🚨 {v}</div>')
                elif "Moderate" in v:
                    _html(f'<div class="alert-card alert-yellow">⚠️ {v}</div>')
                elif "Low bias" in v:
                    _html(f'<div class="alert-card alert-blue">ℹ️ {v}</div>')
                elif "confidence" in v.lower():
                    _html(f'<div class="alert-card alert-blue">🔵 {v}</div>')
                else:
                    _html(f'<div class="alert-card alert-yellow">⚠️ {v}</div>')
        else:
            _html('<hr class="divider">')
            _html('<div class="alert-card alert-green">✅ <b>All governance checks passed</b> — no policy violations detected.</div>')

        # =============================================================
        # 9) SHAP EXPLANATION — clean, professional
        # =============================================================
        _html('<hr class="divider">')
        _html('<div class="section-title">🔍 Why was this decision made?</div>')
        _html('<div class="section-sub">SHAP-based model explanation for this prediction</div>')

        try:
            shap_result = compute_shap_explanation(model, scaler, input_data)

            feature_names = shap_result["feature_names"]
            shap_values   = shap_result["shap_values"]
            base_value    = shap_result["base_value"]
            raw_features  = shap_result["raw_features"]

            # Sort by absolute SHAP value
            sorted_data = sorted(
                zip(feature_names, shap_values, raw_features),
                key=lambda x: abs(x[1]), reverse=True,
            )

            # --- Two-column layout: Chart + Key Insights ---
            col_chart, col_insights = st.columns([3, 2])

            with col_chart:
                # Bar chart — dark themed
                fig, ax = plt.subplots(figsize=(7, 3.2))
                fig.patch.set_facecolor('#0f0f1a')
                ax.set_facecolor('#0f0f1a')

                s_names = [x[0] for x in sorted_data]
                s_vals  = [x[1] for x in sorted_data]
                s_colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in s_vals]

                # Reverse for bottom-to-top (largest at top)
                s_names.reverse(); s_vals.reverse(); s_colors.reverse()

                bars = ax.barh(s_names, s_vals, color=s_colors, height=0.55, edgecolor='none')
                ax.axvline(0, color='#555', linewidth=0.8, linestyle='--')
                ax.set_xlabel("Impact on Approval", fontsize=10, color='#8892b0')
                ax.tick_params(axis='y', labelsize=10, colors='#e0e0ff')
                ax.tick_params(axis='x', labelsize=9, colors='#8892b0')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#333')
                ax.spines['left'].set_color('#333')

                for i, val in enumerate(s_vals):
                    ax.text(
                        val + (0.003 if val >= 0 else -0.003), i,
                        f"{val:+.2f}", va='center',
                        ha='left' if val >= 0 else 'right',
                        fontsize=9, fontweight='bold',
                        color='#e0e0ff',
                    )

                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with col_insights:
                _html('<div class="card"><p style="font-size:14px;font-weight:700;color:#e0e0ff;margin-bottom:14px;">🎯 Key Insights</p>')

                for name, val, raw in sorted_data:
                    if abs(val) < 0.001:
                        continue
                    direction = "positive" if val > 0 else "negative"
                    strength = "Strong" if abs(val) > 0.1 else "Moderate" if abs(val) > 0.03 else "Mild"
                    icon = "🟢" if val > 0 else "🔴"
                    css_class = "impact-pos" if val > 0 else "impact-neg"
                    _html(f"""
                    <div class="insight-chip">
                        <span style="font-size:18px;">{icon}</span>
                        <div>
                            <div class="feat">{name}</div>
                            <div class="impact {css_class}">{strength} {direction} impact</div>
                        </div>
                    </div>
                    """)

                _html('</div>')

            # --- Clean textual explanation ---
            _html('<div class="card" style="margin-top:8px;">')
            _html(f'<p style="font-size:14px;font-weight:700;color:#e0e0ff;margin-bottom:10px;">📝 Plain-English Explanation</p>')

            explanation_parts = []
            for name, val, raw in sorted_data:
                if abs(val) < 0.001:
                    continue
                direction = "increased" if val > 0 else "decreased"
                color = "#2ecc71" if val > 0 else "#e74c3c"
                explanation_parts.append(
                    f'<span style="color:{color};font-weight:600;">{name}</span> {direction} approval chances'
                )

            if explanation_parts:
                _html('<p style="color:#c0c0d0;font-size:14px;line-height:1.8;">' + " · ".join(explanation_parts) + '</p>')
            _html('</div>')

        except Exception as e:
            _html(f'<div class="alert-card alert-yellow">⚠️ SHAP explanation could not be generated: {e}. Prediction and governance results above remain valid.</div>')

        _html('<hr class="divider">')

# ===================================================================
# Logs Section
# ===================================================================
_html('<div class="section-title">📜 Audit & User Logs</div>')
_html('<div class="section-sub">Activity tracking and compliance records</div>')

if AccessControl.check_access(role, "view_logs"):
    tab_user, tab_audit = st.tabs(["👤  User Logs", "📋  Audit Logs"])

    with tab_user:
        user_df = logger.get_user_logs()
        if user_df.empty:
            _html('<div class="alert-card alert-blue">ℹ️ No user logs yet. Run a prediction to generate logs.</div>')
        else:
            st.dataframe(user_df, use_container_width=True, hide_index=True)

    with tab_audit:
        audit_df = logger.get_audit_logs()
        if audit_df.empty:
            _html('<div class="alert-card alert-blue">ℹ️ No audit logs yet.</div>')
        else:
            st.dataframe(audit_df, use_container_width=True, hide_index=True)
else:
    _html('<div class="alert-card alert-red">❌ <b>Unauthorized</b> — your role cannot view logs.</div>')

_html('<hr class="divider">')

# ===================================================================
# Compliance Dashboard
# ===================================================================
_html('<div class="section-title">📈 Compliance Dashboard</div>')
_html('<div class="section-sub">Real-time governance monitoring</div>')

if AccessControl.check_access(role, "view_compliance"):
    # --- summary metrics ---
    audit_df = logger.get_audit_logs()
    if not audit_df.empty:
        audit_df = detect_anomalies(audit_df)

    report = logger.generate_compliance_report()
    if not audit_df.empty and "anomaly" in audit_df.columns:
        report["anomaly_count"] = int(audit_df["anomaly"].sum())

    _html(f"""
    <div class="metric-row">
        <div class="metric-card">
            <div class="label">Total Requests</div>
            <div class="value blue">{report["total_requests"]}</div>
        </div>
        <div class="metric-card">
            <div class="label">Avg Bias Score</div>
            <div class="value {'red' if report['avg_bias_score'] > BIAS_THRESHOLD else 'green'}">{report["avg_bias_score"]}</div>
        </div>
        <div class="metric-card">
            <div class="label">Violations</div>
            <div class="value {'red' if report['total_violations'] > 0 else 'green'}">{report["total_violations"]}</div>
        </div>
        <div class="metric-card">
            <div class="label">Anomalies</div>
            <div class="value {'red' if report['anomaly_count'] > 0 else 'green'}">{report["anomaly_count"]}</div>
        </div>
    </div>
    """)

    # --- fairness metrics ---
    st.markdown("")
    _html('<div class="section-title" style="font-size:16px;">⚖️ Fairness Metrics (Training Data)</div>')
    fairness_full = compute_fairness(df, sensitive_attr="gender")

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Demographic Parity Diff", fairness_full["demographic_parity_diff"])
    fc2.metric("Equalized Odds Diff", fairness_full["equalized_odds_diff"])
    fc3.metric("Bias Detected", "⚠️ Yes" if fairness_full["bias_detected"] else "✅ No")

    # --- anomaly results ---
    if not audit_df.empty and "anomaly" in audit_df.columns:
        anomaly_rows = audit_df[audit_df["anomaly"] == 1]
        _html('<div class="section-title" style="font-size:16px;">🔎 Anomaly Detection</div>')
        if anomaly_rows.empty:
            _html('<div class="alert-card alert-green">✅ No anomalies detected in audit logs.</div>')
        else:
            _html(f'<div class="alert-card alert-yellow">🔍 {len(anomaly_rows)} anomalous record(s) found.</div>')
            st.dataframe(anomaly_rows, use_container_width=True, hide_index=True)

    # --- violations summary ---
    if not audit_df.empty:
        violations_df = audit_df[audit_df["policy_violations"] != "None"]
        _html('<div class="section-title" style="font-size:16px;">🚫 Violations Summary</div>')
        if violations_df.empty:
            _html('<div class="alert-card alert-green">✅ No policy violations recorded.</div>')
        else:
            st.dataframe(violations_df[["timestamp", "decision", "policy_violations"]], use_container_width=True, hide_index=True)
else:
    _html('<div class="alert-card alert-red">❌ <b>Unauthorized</b> — your role cannot view the compliance dashboard.</div>')

# Footer
_html("""
<hr class="divider">
<div style="text-align:center;padding:10px 0 20px 0;">
    <p style="color:#555;font-size:12px;">Responsible AI Governance Framework · Built with Streamlit · SHAP Explainability</p>
</div>
""")
