import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False

# =========================
# 语言选择
# =========================
lang = st.sidebar.selectbox("Language / 语言", ["English", "中文"])

text = {
    "English": {
        "title": "Prediction Tool for Nosocomial Infections in ACLF",
        "binary_title": "Binary Features (Yes/No)",
        "numeric_title": "Numerical Features",
        "predict_button": "Predict",
        "infection_prob": "Probability of Infection",
        "risk_result": "Risk Assessment",
        "high": "High Risk",
        "low": "Low Risk",
        "threshold": "Threshold",
        "disclaimer": "Disclaimer: For research use only.",
        "feature_labels": {
            "Antibiotic use": "Antibiotic Use",
            "Circulatory Failure": "Circulatory Failure",
            "HE": "Hepatic Encephalopathy",
            "Alb": "Albumin (g/L)",
            "WBC": "White Blood Cells (×10⁹/L)",
            "INR": "INR",
            "Cr": "Creatinine (µmol/L)",
            "CRP": "C-reactive Protein (mg/L)",
            "LDL-C": "LDL-C (mmol/L)"
        }
    },
    "中文": {
        "title": "ACLF院内感染风险预测工具",
        "binary_title": "二分类特征（是/否）",
        "numeric_title": "数值型特征",
        "predict_button": "预测",
        "infection_prob": "院内感染概率",
        "risk_result": "风险评估",
        "high": "高风险",
        "low": "低风险",
        "threshold": "阈值",
        "disclaimer": "免责声明：仅供科研参考。",
        "feature_labels": {
            "Antibiotic use": "抗生素使用",
            "Circulatory Failure": "循环衰竭",
            "HE": "肝性脑病",
            "Alb": "白蛋白 (g/L)",
            "WBC": "白细胞 (×10⁹/L)",
            "INR": "INR",
            "Cr": "肌酐 (µmol/L)",
            "CRP": "C反应蛋白 (mg/L)",
            "LDL-C": "低密度脂蛋白 (mmol/L)"
        }
    }
}

t = text[lang]

# =========================
# 加载模型
# =========================
model = joblib.load("XGBmodel.pkl")

feature_names = [
    'Antibiotic use',
    'Circulatory Failure',
    'HE',
    'Alb',
    'WBC',
    'INR',
    'Cr',
    'CRP',
    'LDL-C'
]

# =========================
# 页面标题
# =========================
st.markdown(f"<h1 style='text-align: center;'>{t['title']}</h1>", unsafe_allow_html=True)

# =========================
# 输入界面
# =========================
user_input = {}

binary_features = ['Antibiotic use', 'Circulatory Failure', 'HE']
st.subheader(t["binary_title"])

for feature in binary_features:
    label = t["feature_labels"][feature]
    choice = st.selectbox(label, ["No", "Yes"] if lang=="English" else ["否","是"])
    user_input[feature] = 1 if choice in ["Yes","是"] else 0

numeric_features = ['Alb','WBC','INR','Cr','CRP','LDL-C']
default_values = {'Alb':30,'WBC':6,'INR':1.2,'Cr':70,'CRP':20,'LDL-C':2.5}

st.subheader(t["numeric_title"])

for feature in numeric_features:
    label = t["feature_labels"][feature]
    val = st.number_input(label, value=float(default_values[feature]))
    user_input[feature] = val

# =========================
# 预测
# =========================
if st.button(t["predict_button"]):

    input_array = np.array([[user_input[f] for f in feature_names]])
    prob = model.predict_proba(input_array)[0][1]

    st.write(f"**{t['infection_prob']}**: {prob*100:.1f}%")

    # 约登指数最佳cutoff（替换成你真实计算值）
    threshold = 0.404

    if prob >= threshold:
        st.error(f"{t['risk_result']} ({t['threshold']} {threshold}): {t['high']}")
    else:
        st.success(f"{t['risk_result']} ({t['threshold']} {threshold}): {t['low']}")

    st.info(t["disclaimer"])

    # =========================
    # SHAP解释
    # =========================
    if st.button("Show SHAP Plot" if lang=="English" else "显示SHAP图"):

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(input_array)

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        base_value = explainer.expected_value[1] if isinstance(explainer.expected_value,list) else explainer.expected_value

        plt.figure(figsize=(10,6))
        shap.force_plot(
            base_value,
            shap_values[0],
            input_array[0],
            feature_names=feature_names,
            matplotlib=True,
            show=False
        )

        st.pyplot(plt.gcf())
