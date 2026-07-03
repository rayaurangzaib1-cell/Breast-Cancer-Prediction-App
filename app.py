"""
Bio-Scan AI Diagnostic Laboratory
---------------------------------
Professional clinical-style Streamlit front-end for a breast mass
(FNA) classification model trained on the Wisconsin Diagnostic
Breast Cancer feature set.

Developed by Aurang Zeb
"""

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.datasets import load_breast_cancer

# ======================================================================
# PAGE CONFIG
# ======================================================================
st.set_page_config(
    page_title="Bio-Scan AI Diagnostic Laboratory",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ======================================================================
# CUSTOM CSS — clinical / lab theme
# ======================================================================
st.markdown("""
<style>
    :root{
        --lab-navy:#0b2545;
        --lab-teal:#0f766e;
        --lab-red:#b91c1c;
        --lab-green:#15803d;
        --lab-bg:#f4f7f9;
    }
    .stApp{ background-color: var(--lab-bg); }

    /* Header banner */
    .lab-header{
        background: linear-gradient(90deg, var(--lab-navy) 0%, #123a63 100%);
        padding: 22px 30px;
        border-radius: 10px;
        color: white;
        margin-bottom: 18px;
        box-shadow: 0 4px 14px rgba(0,0,0,0.15);
    }
    .lab-header h1{
        margin:0; font-size: 30px; font-weight:700; letter-spacing:0.3px;
    }
    .lab-header p{
        margin:4px 0 0 0; font-size:14px; color:#cfe3f5; letter-spacing:0.5px;
    }
    .dev-credit{
        text-align:right; color:#dbe9f7; font-size:13px; line-height:1.4;
    }
    .dev-credit b{ color:#ffffff; font-size:14px; }

    /* Section titles */
    .section-title{
        font-size:19px; font-weight:700; color: var(--lab-navy);
        border-left: 5px solid var(--lab-teal);
        padding-left:10px; margin: 18px 0 10px 0;
    }

    /* Report card */
    .report-card{
        background:#ffffff; border:1px solid #d9e2ec; border-radius:10px;
        padding:26px 30px; box-shadow:0 2px 10px rgba(0,0,0,0.06);
    }
    .report-letterhead{
        display:flex; justify-content:space-between; align-items:center;
        border-bottom:2px solid var(--lab-navy); padding-bottom:12px; margin-bottom:16px;
    }
    .report-letterhead h2{ margin:0; color:var(--lab-navy); font-size:22px;}
    .report-letterhead span{ color:#5b6b7b; font-size:12.5px;}

    .patient-grid{
        display:grid; grid-template-columns: repeat(4, 1fr); gap:10px 24px;
        background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px;
        padding:14px 18px; margin-bottom:18px; font-size:14px;
    }
    .patient-grid div b{ color:var(--lab-navy); }

    .verdict-box{
        border-radius:10px; padding:16px 20px; margin:16px 0;
        font-size:18px; font-weight:700; text-align:center;
        letter-spacing:0.4px;
    }
    .verdict-malignant{ background:#fef2f2; border:2px solid var(--lab-red); color:var(--lab-red); }
    .verdict-benign{ background:#f0fdf4; border:2px solid var(--lab-green); color:var(--lab-green); }

    .report-footer{
        margin-top:22px; padding-top:12px; border-top:1px dashed #cbd5e1;
        font-size:12px; color:#64748b; display:flex; justify-content:space-between;
    }

    .status-abnormal{ color:var(--lab-red); font-weight:700; }
    .status-normal{ color:var(--lab-green); font-weight:600; }

    .stButton>button{
        background: var(--lab-teal); color:white; font-weight:600;
        border-radius:8px; padding:10px 18px; border:none;
    }
    .stButton>button:hover{ background: var(--lab-navy); color:white; }
</style>
""", unsafe_allow_html=True)

# ======================================================================
# FEATURE DEFINITIONS (order MUST match training order)
# ======================================================================
FEATURE_NAMES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry", "mean fractal dimension",
    "radius error", "texture error", "perimeter error", "area error", "smoothness error",
    "compactness error", "concavity error", "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area", "worst smoothness",
    "worst compactness", "worst concavity", "worst concave points", "worst symmetry", "worst fractal dimension"
]

GROUP_MEAN = FEATURE_NAMES[0:10]
GROUP_ERROR = FEATURE_NAMES[10:20]
GROUP_WORST = FEATURE_NAMES[20:30]


def pretty(name: str) -> str:
    return name.title().replace("Se ", "SE ")


# ======================================================================
# CACHED LOADERS
# ======================================================================
@st.cache_resource(show_spinner="Loading trained model...")
def load_artifacts():
    model = joblib.load("breast_cancer_model.pkl")
    scaler = joblib.load("scaler.pkl")
    return model, scaler


@st.cache_data(show_spinner=False)
def get_reference_ranges():
    """
    Builds clinical 'reference ranges' from the benign-class distribution
    of the Wisconsin Diagnostic Breast Cancer dataset (same feature set
    used to train the model). Range = mean ± 1 standard deviation.
    Returns dict: feature -> (low, high, mean)
    """
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    benign = df[df["target"] == 1]  # sklearn: 0=malignant, 1=benign
    ranges = {}
    for col in data.feature_names:
        m, s = benign[col].mean(), benign[col].std()
        ranges[col] = (m - s, m + s, m)
    return ranges


try:
    model, scaler = load_artifacts()
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    LOAD_ERROR = str(e)

REF_RANGES = get_reference_ranges()

# ======================================================================
# HEADER
# ======================================================================
h1, h2 = st.columns([3.2, 1])
with h1:
    st.markdown("""
    <div class="lab-header">
        <h1>🧬 Bio-Scan AI Diagnostic Laboratory</h1>
        <p>Computational Cytopathology &nbsp;•&nbsp; Fine Needle Aspirate (FNA) Analysis &nbsp;•&nbsp; AI-Assisted Screening</p>
    </div>
    """, unsafe_allow_html=True)
with h2:
    st.markdown(f"""
    <div class="lab-header dev-credit">
        <b>Developed by</b><br>Aurang Zeb<br>
        <span style="font-size:11px;">v1.0 &nbsp;|&nbsp; {datetime.now().strftime('%Y')}</span>
    </div>
    """, unsafe_allow_html=True)

if not MODEL_LOADED:
    st.error(
        f"⚠️ Could not load model/scaler files (`breast_cancer_model.pkl`, "
        f"`scaler.pkl`). Make sure both files are in the app's root directory.\n\n"
        f"Details: {LOAD_ERROR}"
    )
    st.stop()

# ======================================================================
# SIDEBAR — Patient Registration + User Guide
# ======================================================================
with st.sidebar:
    st.markdown("## 🧾 Patient Registration")
    p_name = st.text_input("Full Name", "John Doe")
    c1, c2 = st.columns(2)
    with c1:
        p_age = st.number_input("Age", min_value=1, max_value=120, value=35)
    with c2:
        p_gender = st.selectbox("Gender", ["Female", "Male", "Other"])
    p_id = st.text_input("Patient ID / MRN", "BCA-" + datetime.now().strftime("%y%m%d"))
    ref_by = st.text_input("Referring Physician", "Dr. —")
    sample_id = st.text_input("Sample / Specimen ID", "FNA-0001")

    st.divider()
    with st.expander("📖 User Guide — How to use this app", expanded=False):
        st.markdown("""
        **Step 1 — Patient Details**
        Fill in patient name, age, gender and ID in this sidebar.

        **Step 2 — Enter Parameters**
        Go to the *Mean*, *Standard Error* and *Worst* tabs and enter the
        30 cytological measurements obtained from the digitized FNA image
        (radius, texture, perimeter, area, smoothness, etc.).
        Hover the ℹ️ icon on each field to see its typical **benign
        reference range**.

        **Step 3 — Generate Report**
        Click **"Generate Diagnostic Report"**. The AI model will classify
        the sample and produce a formatted lab report, flagging any
        parameter that falls **outside the normal (benign) reference
        range**.

        **Step 4 — Export**
        Use the download buttons to save the report as a text file or a
        printable HTML file (open it in a browser and press *Ctrl+P* to
        print or save as PDF).

        ⚠️ **Disclaimer:** This tool is an educational / decision-support
        demo. It does **not** replace histopathological confirmation by a
        licensed pathologist.
        """)

    st.divider()
    st.caption("⚠️ For clinical decision-support and educational use only. "
               "Always confirm results with a licensed medical professional.")

# ======================================================================
# PARAMETER ENTRY
# ======================================================================
st.markdown('<div class="section-title">📋 Cytological / Morphometric Parameters</div>', unsafe_allow_html=True)
st.caption("Enter the 30 quantitative features extracted from the digitized FNA image.")

tab_mean, tab_err, tab_worst = st.tabs(
    ["🔹 Mean Values", "📉 Standard Error", "🔺 Worst (Largest) Values"]
)

inputs = {}


def render_group(tab, group_features):
    with tab:
        cols = st.columns(3)
        for i, feat in enumerate(group_features):
            lo, hi, _ = REF_RANGES[feat]
            inputs[feat] = cols[i % 3].number_input(
                pretty(feat),
                format="%.4f",
                value=0.0,
                help=f"Typical benign reference range ≈ {lo:.3f} – {hi:.3f}",
                key=f"in_{feat}",
            )


render_group(tab_mean, GROUP_MEAN)
render_group(tab_err, GROUP_ERROR)
render_group(tab_worst, GROUP_WORST)

st.markdown("<br>", unsafe_allow_html=True)
generate = st.button("🔬 Generate Diagnostic Report", type="primary", use_container_width=True)

# ======================================================================
# REPORT GENERATION
# ======================================================================
if generate:
    values = [inputs[f] for f in FEATURE_NAMES]
    arr = np.array(values).reshape(1, -1)
    scaled = scaler.transform(arr)
    prediction = model.predict(scaled)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        p = model.predict_proba(scaled)[0]
        # class 0 = malignant, class 1 = benign (sklearn convention)
        proba = {"malignant": p[0], "benign": p[1]}

    is_malignant = (prediction == 0)

    # ---- Build parameter table with normal/abnormal flags ----
    rows, abnormal_count = [], 0
    for feat, val in zip(FEATURE_NAMES, values):
        lo, hi, _ = REF_RANGES[feat]
        status = "Normal"
        if val < lo or val > hi:
            status = "Abnormal"
            abnormal_count += 1
        rows.append({
            "Parameter": pretty(feat),
            "Patient Value": f"{val:.4f}",
            "Reference Range (Benign)": f"{lo:.3f} – {hi:.3f}",
            "Status": status,
        })
    report_df = pd.DataFrame(rows)

    st.markdown("---")
    st.markdown('<div class="section-title">📄 Official Diagnostic Report</div>', unsafe_allow_html=True)

    now_str = datetime.now().strftime("%d %B %Y, %H:%M:%S")
    verdict_class = "verdict-malignant" if is_malignant else "verdict-benign"
    verdict_icon = "🚨" if is_malignant else "✅"
    verdict_text = "MALIGNANT — Positive Findings (Further Biopsy Recommended)" \
        if is_malignant else "BENIGN — Negative Findings (No Malignancy Detected)"

    confidence_line = ""
    if proba:
        conf = proba["malignant"] if is_malignant else proba["benign"]
        confidence_line = f"<div style='text-align:center; color:#5b6b7b; font-size:13px;'>Model confidence: {conf*100:.1f}%</div>"

    st.markdown(f"""
    <div class="report-card">
        <div class="report-letterhead">
            <div>
                <h2>🧬 Bio-Scan AI Diagnostic Laboratory</h2>
                <span>Computational Pathology Report — FNA Cytology Panel</span>
            </div>
            <div style="text-align:right; font-size:12.5px; color:#5b6b7b;">
                Report generated: {now_str}<br>
                Report ID: RPT-{datetime.now().strftime('%y%m%d%H%M%S')}
            </div>
        </div>

        <div class="patient-grid">
            <div><b>Patient Name:</b> {p_name}</div>
            <div><b>Age / Gender:</b> {p_age} / {p_gender}</div>
            <div><b>Patient ID:</b> {p_id}</div>
            <div><b>Sample ID:</b> {sample_id}</div>
            <div><b>Referring Physician:</b> {ref_by}</div>
            <div><b>Total Parameters:</b> 30</div>
            <div><b>Abnormal Findings:</b> {abnormal_count}</div>
            <div><b>Analysis Method:</b> AI Classifier (ML)</div>
        </div>

        <div class="verdict-box {verdict_class}">{verdict_icon} {verdict_text}</div>
        {confidence_line}
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("**Detailed Parameter Analysis**")

    def highlight_status(row):
        color = "#b91c1c" if row["Status"] == "Abnormal" else "#15803d"
        weight = "700" if row["Status"] == "Abnormal" else "500"
        return [f"color:{color}; font-weight:{weight};" if col == "Status" else "" for col in row.index]

    st.dataframe(
        report_df.style.apply(highlight_status, axis=1),
        use_container_width=True,
        hide_index=True,
        height=420,
    )

    st.markdown(f"""
    <div class="report-card" style="margin-top:14px;">
        <p style="font-size:13.5px; color:#334155;">
        <b>Clinical Note:</b> Parameters marked <span class="status-abnormal">Abnormal</span>
        fall outside the typical benign reference range (mean ± 1 SD, derived from the
        Wisconsin Diagnostic Breast Cancer reference dataset) and may indicate malignant
        morphological characteristics. This automated result should be correlated with
        histopathological examination and clinical findings before any diagnostic or
        treatment decision is made.
        </p>
        <div class="report-footer">
            <span>Developed by <b>Aurang Zeb</b> — Bio-Scan AI Diagnostic Laboratory</span>
            <span>This is an AI-generated screening report, not a certified medical diagnosis.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ---- Downloads ----
    text_report = f"""BIO-SCAN AI DIAGNOSTIC LABORATORY
=========================================
Report generated : {now_str}
Report ID         : RPT-{datetime.now().strftime('%y%m%d%H%M%S')}
-----------------------------------------
Patient Name      : {p_name}
Age / Gender      : {p_age} / {p_gender}
Patient ID        : {p_id}
Sample ID         : {sample_id}
Referring Physician: {ref_by}
-----------------------------------------
DIAGNOSIS RESULT  : {verdict_text}
Abnormal Findings : {abnormal_count} / 30 parameters
-----------------------------------------
PARAMETER DETAILS
-----------------------------------------
""" + "\n".join(
        f"{r['Parameter']:<28} Value: {r['Patient Value']:<10} "
        f"Ref: {r['Reference Range (Benign)']:<20} Status: {r['Status']}"
        for r in rows
    ) + """
-----------------------------------------
Note: This is an AI-generated screening report and does not replace
a certified pathologist's diagnosis. Please consult a specialist.

Developed by Aurang Zeb — Bio-Scan AI Diagnostic Laboratory
"""

    html_report = f"""
    <html><head><meta charset="utf-8"><title>Bio-Scan Report - {p_name}</title>
    <style>
        body{{font-family:Arial, sans-serif; padding:30px; color:#0b2545;}}
        h1{{color:#0b2545;}} table{{border-collapse:collapse; width:100%; margin-top:14px;}}
        th,td{{border:1px solid #cbd5e1; padding:6px 10px; font-size:13px; text-align:left;}}
        th{{background:#0b2545; color:white;}}
        .abn{{color:#b91c1c; font-weight:bold;}} .norm{{color:#15803d;}}
        .verdict{{padding:12px; border-radius:8px; font-weight:bold; font-size:16px; text-align:center;
                  border:2px solid {'#b91c1c' if is_malignant else '#15803d'};
                  color:{'#b91c1c' if is_malignant else '#15803d'};
                  background:{'#fef2f2' if is_malignant else '#f0fdf4'};}}
    </style></head><body>
    <h1>🧬 Bio-Scan AI Diagnostic Laboratory</h1>
    <p>Report generated: {now_str} &nbsp;|&nbsp; Report ID: RPT-{datetime.now().strftime('%y%m%d%H%M%S')}</p>
    <p><b>Patient:</b> {p_name} &nbsp; | &nbsp; <b>Age/Gender:</b> {p_age}/{p_gender} &nbsp; | &nbsp;
       <b>ID:</b> {p_id} &nbsp; | &nbsp; <b>Sample:</b> {sample_id} &nbsp; | &nbsp;
       <b>Referring Physician:</b> {ref_by}</p>
    <div class="verdict">{verdict_text}</div>
    <table><tr><th>Parameter</th><th>Patient Value</th><th>Reference Range</th><th>Status</th></tr>
    {''.join(f"<tr><td>{r['Parameter']}</td><td>{r['Patient Value']}</td><td>{r['Reference Range (Benign)']}</td><td class='{'abn' if r['Status']=='Abnormal' else 'norm'}'>{r['Status']}</td></tr>" for r in rows)}
    </table>
    <p style="margin-top:20px; font-size:12px; color:#64748b;">
    This is an AI-generated screening report and does not replace a certified pathologist's diagnosis.<br>
    Developed by <b>Aurang Zeb</b> — Bio-Scan AI Diagnostic Laboratory
    </p>
    </body></html>
    """

    d1, d2 = st.columns(2)
    with d1:
        st.download_button(
            "📥 Download Report (.txt)",
            text_report,
            file_name=f"Report_{p_name.replace(' ', '_')}.txt",
            use_container_width=True,
        )
    with d2:
        st.download_button(
            "🖨️ Download Printable Report (.html)",
            html_report,
            file_name=f"Report_{p_name.replace(' ', '_')}.html",
            mime="text/html",
            use_container_width=True,
        )
else:
    st.info("Fill in the parameters above and click **Generate Diagnostic Report** to produce the lab report.")
