# 🧠 Introvert vs. Extrovert Prediction

A machine learning classification project that predicts whether a person is more likely to be **Introverted** or **Extroverted** based on behavioral and social traits.

## 📌 Project Overview

The project includes:

- Exploratory Data Analysis
- Missing-value analysis
- Feature engineering
- Model comparison
- Hyperparameter tuning with Optuna
- Interactive Streamlit app

## 📊 Dataset

The notebook uses **18,524 training records** with features such as:

- Time spent alone
- Stage fear
- Social event attendance
- Going outside
- Feeling drained after socializing
- Friends circle size
- Social media post frequency

Target:

```text id="rn3h01"
Personality
```

## 🤖 Model Comparison

The notebook compares:

```text id="9x0h04"
XGBoost  : 96.89%
LightGBM : 96.92%
CatBoost : 96.89%
```

LightGBM is also tuned with **Optuna**, reaching approximately **96.93% accuracy**.

## 🖥️ Streamlit App

The `introextro.py` application uses **CatBoost** to predict personality type.

The app displays:

- Introvert / Extrovert prediction
- Prediction confidence
- Class probabilities
- Basic EDA charts

Run the app:

```bash id="ymf7oi"
streamlit run introextro.py
```

> Note: The Streamlit app currently trains on a synthetic dataset generated inside the application.

## 🛠️ Technologies

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- XGBoost
- LightGBM
- CatBoost
- Optuna
- Streamlit
- Jupyter Notebook

## 📁 Project Structure

```text id="5lwjm4"
introvert-extrovert-classification/
├── IntrovertExtrovert.ipynb
├── introextro.py
├── train.csv
├── test.csv
└── README.md
```

## 🎯 Skills Demonstrated

- Exploratory Data Analysis
- Feature engineering
- Classification
- Cross-validation
- Hyperparameter tuning
- Boosting models
- Streamlit development

---

Built with Python, LightGBM, CatBoost, XGBoost, and Streamlit. 🧠🤖
