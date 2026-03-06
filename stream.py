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
        "disclaimer": "Disclaimer: This result is for reference only and should not be used for diagnosis or treatment decisions.",
        "feature_labels": {
            "Antibiotic use": "Antibiotic Use",
            "Circulatory Failure": "Circulatory Failure",
            "HE": "Hepatic Encephalopathy",
            "ALB": "Albumin (g/L)",
            "WBC": "White Blood Cells (×10⁹/L)",
            "INR": "INR",
            "Cr": "Creatinine (µmol/L)",
            "CRP": "C-reactive Protein (mg/L)",
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
        "disclaimer": "免责声明：本结果仅供参考，不可作为诊断或治疗决策依据。",
        "feature_labels": {
            "Antibiotic use": "抗生素使用",
            "Circulatory Failure": "循环衰竭",
            "HE": "肝性脑病",
            "ALB": "白蛋白 (g/L)",
            "WBC": "白细胞 (×10⁹/L)",
            "INR": "INR",
            "Cr": "肌酐 (µmol/L)",
            "CRP": "C反应蛋白 (mg/L)",
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
    'ALB',
    'WBC',
    'INR',
    'Cr',
    'CRP',
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

numeric_features = ['ALB','WBC','INR','Cr','CRP']
default_values = {'ALB':30,'WBC':6,'INR':1.2,'Cr':70,'CRP':20}

st.subheader(t["numeric_title"])

for feature in numeric_features:
    label = t["feature_labels"][feature]
    val = st.number_input(label, value=float(default_values[feature]))
    user_input[feature] = val

# =========================
# 预测
# =========================
if st.button(t["predict_button"]):
    # 按训练模型顺序构造特征
    features = np.array([[user_input[f] for f in feature_names]], dtype=float)

    predicted_proba = model.predict_proba(features)[0]
    class1_prob = predicted_proba[1] * 100

    st.write(f"**{t['infection_prob']}：** {class1_prob:.1f}%")

    threshold = 0.402
    risk = t["high"] if class1_prob/100 >= threshold else t["low"]
    st.write(f"**{t['risk_result']}（{t['threshold']} {threshold:.3f}）：** {risk}")

    st.info(t["disclaimer"])
# 5️⃣ SHAP 可解释性可视化（点击按钮显示）
# =========================
import matplotlib.pyplot as plt
import shap
import pandas as pd
import streamlit as st

# =========================
show_shap = st.button("Show SHAP Force Plot" if lang == "English" else "显示 SHAP 图")

if show_shap:
    # 构建输入数据
    input_df = pd.DataFrame([list(user_input.values())], columns=feature_names)
    
    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(input_df)
    shap_values_for_sample = sv[1][0] if isinstance(sv, list) else sv[0]
    
    # 基准值和模型输出 f(x)
    base_value = explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value
    fx_value = base_value + shap_values_for_sample.sum()
    
    # 绘图
    plt.figure(figsize=(12, 10))
    shap.force_plot(
        base_value,
        shap_values_for_sample,
        input_df.iloc[0],
        feature_names=feature_names,
        matplotlib=True,
        show=False
    )
    
    ax = plt.gca()
    
    # base value 虚线和数值
    ax.axvline(base_value, color='gray', linestyle='--', linewidth=1)
    ax.text(base_value, ax.get_ylim()[1]*1.05,
            f'{base_value:.3f}', color='gray', fontsize=12,
            ha='center', va='bottom', fontweight='bold')
    
    # 调整字体大小
    for label in ax.get_yticklabels():
        label.set_fontsize(14)
    for label in ax.get_xticklabels():
        label.set_fontsize(14)
    for text in ax.texts:
        text.set_fontsize(11)
    
    plt.tight_layout()
    st.pyplot(plt.gcf())

    # =========================
    # 力图说明
    # =========================
    if lang == "中文":
        with st.expander("🧩 点击查看 SHAP 力图详细解释"):
            st.markdown("""
**SHAP 力图（SHAP Force Plot）** 用于解释单个样本的预测结果，展示每个特征对模型输出的影响。

**1️⃣ 基线值（Base Value）**  
- 图中标记的 *base value* 表示模型的平均输出。  

**2️⃣ 模型输出值（f(x)）**  
- 图中显示的 *f(x)* 值是该样本的最终预测结果。  
- 它等于基线值加上所有特征的 SHAP 值：  
  `f(x) = base value + Σ(SHAP_i)`  

**3️⃣ 特征贡献（红色和蓝色箭头）**  
- 🔴 **红色箭头**：对预测结果有正向贡献（推高预测值）。  
- 🔵 **蓝色箭头**：对预测结果有负向贡献（降低预测值）。  

**4️⃣ 影响程度（箭头长度）**  
- 箭头越长，说明该特征的 SHAP 值绝对值越大，对当前样本预测的影响越显著。  

**📘 总结**  
- 左侧（蓝色）特征使模型预测值减小；  
- 右侧（红色）特征使预测值增大；  
- 中间的灰色虚线表示模型的平均预测水平。
""")
    else:
        with st.expander("🧩 Click to view detailed SHAP Force Plot explanation"):
            st.markdown("""
**SHAP Force Plot** is used to interpret the prediction of an individual sample by showing how each feature contributes to the model output.

**1️⃣ Base Value**  
- The *base value* represents the model’s average output.  

**2️⃣ Model Output (f(x))**  
- The *f(x)* indicates the final predicted value for this sample.  
- It equals the base value plus the sum of all SHAP values:  
  `f(x) = base value + Σ(SHAP_i)`  

**3️⃣ Feature Contributions (Red and Blue Arrows)**  
- 🔴 **Red arrows**: Features that push the prediction higher (positive contribution).  
- 🔵 **Blue arrows**: Features that push the prediction lower (negative contribution).  

**4️⃣ Magnitude of Impact (Arrow Length)**  
- Longer arrows indicate features with larger absolute SHAP values, meaning stronger influence on the prediction.  

**📘 Summary**  
- Features on the **left (blue)** decrease the predicted value;  
- Features on the **right (red)** increase it;  
- The **gray dashed line** represents the model’s average output.
""")

