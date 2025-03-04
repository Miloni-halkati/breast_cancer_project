# 🩺 Breast Cancer Prediction App  

This **Machine Learning-powered web application** predicts whether a tumor is **Malignant** or **Benign** based on input features.

🔗 **Live App:** [Breast Cancer Prediction](https://breastcancerproject-mwfrp6euxvrsijeczh4fqt.streamlit.app/)

---

## 📊 Dataset Used  

The model is trained using the **Breast Cancer Wisconsin Dataset** from the **UCI Machine Learning Repository**.  
This dataset contains **30 numeric features** extracted from **digitized images of breast cancer cells**.  

✅ **Classes:**  
- `Malignant (1)` → Cancerous  
- `Benign (0)` → Non-Cancerous  

✅ **Features (Example)**:
- **Radius (mean, worst, standard error)**
- **Texture (mean, worst, standard error)**
- **Perimeter (mean, worst, standard error)**
- **Area, Compactness, Concavity, Symmetry, Fractal Dimension**

📌 **For more details, check the dataset source:** [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29)

---

## 📈 Summary Statistics of Dataset  

### **Mean Values**  
| Feature | Min | Max |
|----------|------|------|
| Radius (mean) | 6.981 | 28.11 |
| Texture (mean) | 9.71 | 39.28 |
| Perimeter (mean) | 43.79 | 188.5 |
| Area (mean) | 143.5 | 2501.0 |
| Smoothness (mean) | 0.053 | 0.163 |
| Compactness (mean) | 0.019 | 0.345 |
| Concavity (mean) | 0.0 | 0.427 |
| Concave points (mean) | 0.0 | 0.201 |
| Symmetry (mean) | 0.106 | 0.304 |
| Fractal dimension (mean) | 0.05 | 0.097 |

### **Standard Error & Worst Case Values**  
| Feature | Standard Error (Min-Max) | Worst (Min-Max) |
|----------|----------------|----------------|
| Radius | 0.112 - 2.873 | 7.93 - 36.04 |
| Texture | 0.36 - 4.885 | 12.02 - 49.54 |
| Perimeter | 0.757 - 21.98 | 50.41 - 251.2 |
| Area | 6.802 - 542.2 | 185.2 - 4254.0 |
| Smoothness | 0.002 - 0.031 | 0.071 - 0.223 |
| Compactness | 0.002 - 0.135 | 0.027 - 1.058 |
| Concavity | 0.0 - 0.396 | 0.0 - 1.252 |
| Concave Points | 0.0 - 0.053 | 0.0 - 0.291 |
| Symmetry | 0.008 - 0.079 | 0.156 - 0.664 |
| Fractal Dimension | 0.001 - 0.03 | 0.055 - 0.208 |

---

## 🚀 How to Use the App  
1️⃣ Open the **[Live App](https://breastcancerproject-mwfrp6euxvrsijeczh4fqt.streamlit.app/)**  
2️⃣ Enter **30 feature values** (comma-separated)  
3️⃣ Click **"Predict"**  
4️⃣ View **Prediction Result** (Benign/Malignant with probability)  

---

## 🔬 Technologies Used  
- **Python**  
- **PyTorch** (Deep Learning)  
- **Scikit-Learn** (Data Preprocessing)  
- **Streamlit** (Web UI)  
- **Git & GitHub** (Version Control)  

---

## 🖥️ Run the Project Locally  
### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Miloni-halkati/breast_cancer_project.git
cd breast_cancer_project
