# 🩺 GlucoGuide

**AI-Based Diabetes Prediction and Personalized Risk Analysis**

GlucoGuide is a premium, production-ready Streamlit application that uses a
pre-trained XGBoost multiclass classifier to predict a patient's diabetes
status from 11 clinical indicators, and presents a personalized, visually
rich risk report.

---

## ⚠️ Important — Model Note

This app is built around the **actual trained artifacts provided**, which
differ from a "classic" binary Pima-Indians-style model:

- **Model type:** `XGBClassifier` — **5-class** multiclass classifier (not binary)
- **Predicted classes:** `No Diabetes`, `Pre-Diabetes`, `Type 1`, `Type 2`, `Gestational`
- **Input features (11):** `age`, `gender`, `bmi`, `hba1c`, `glucose_fasting`,
  `glucose_postprandial`, `cholesterol_total`, `systolic_bp`,
  `family_history_diabetes`, `physical_activity_minutes_per_week`, `insulin_level`
- **Preprocessing:** a `ColumnTransformer` (`StandardScaler` for numeric
  features + `OneHotEncoder` for `gender`) saved as `preprocessor.pkl`

If you intended to use the classic 8-feature Pima dataset (Pregnancies,
Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction,
Age) with a binary output, you'll need to retrain and re-export a model with
that schema — the current `.pkl` files do not match that structure.

---

## 📁 Project Structure

```
project/
│
├── app.py                  # Main Streamlit application
├── models/
│      xgb_model.pkl        # Trained XGBoost classifier
│      preprocessor.pkl     # ColumnTransformer (scaler + one-hot encoder)
│      label_encoder.pkl    # LabelEncoder for the 5 target classes
├── requirements.txt
└── README.md
```

## 🚀 Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🧠 Model Details

| Property        | Value                                   |
|-----------------|------------------------------------------|
| Algorithm       | XGBoost Classifier                       |
| Task            | Multiclass classification (5 classes)    |
| Preprocessing   | StandardScaler + OneHotEncoder (gender)  |
| Output          | Predicted class + full probability vector|

## 🛠️ Tech Stack

- **Python 3.10+**
- **Streamlit** — UI framework
- **XGBoost / scikit-learn** — model & preprocessing
- **Plotly** — gauge & probability charts
- **Pandas** — data handling

## ⚕️ Medical Disclaimer

GlucoGuide is an AI-assisted screening tool intended for educational and
informational purposes only. It does **not** provide a medical diagnosis.
Always consult a licensed healthcare professional for diagnosis and treatment.
