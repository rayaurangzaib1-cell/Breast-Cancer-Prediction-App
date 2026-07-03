# 🧬 Bio-Scan AI Diagnostic Laboratory

Professional, lab-report–style Streamlit interface for a breast-mass (FNA)
classification model. Developed by **Aurang Zeb**.

## Features
- Clinical letterhead-style UI with lab branding
- Patient registration panel (name, age, gender, ID, referring physician)
- 30 parameters grouped into Mean / Standard Error / Worst tabs
- Reference ("normal") ranges computed live from the benign class of the
  Wisconsin Diagnostic Breast Cancer dataset (mean ± 1 SD)
- Auto-flags **Abnormal** parameters in the generated report
- Printable HTML report + plain-text report, both downloadable
- Built-in sidebar user guide

## Files needed in the repo root
```
app.py
requirements.txt
breast_cancer_model.pkl   # your trained model
scaler.pkl                 # your fitted scaler
```

> The app expects the model to have been trained on the standard
> `sklearn.datasets.load_breast_cancer()` feature order (30 features) and
> the scaler fitted on the same features.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Push `app.py`, `requirements.txt`, `breast_cancer_model.pkl`, `scaler.pkl`
   to a public GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), connect the repo,
   set `app.py` as the entry point, and deploy.

## Disclaimer
This tool is for educational / decision-support demonstration only. It is
**not** a certified diagnostic device and does not replace evaluation by a
licensed pathologist.
