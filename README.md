# 🧠 Mental Health in Tech — ML Prediction System

> *"Patterns come from data; solutions come from empathy."*

A machine learning project that predicts whether a tech employee may need mental health support or medical treatment, based on workplace survey data.

---

## 📌 Problem Statement

Employee mental stress is often undetected until it's too late — leading to reduced productivity, poor well-being, and high attrition. This project builds an automated system to identify stress risk early using real employee survey data.

**The cost of late detection:**
- 💸 ~$5M/year average loss for a 1,000-employee company due to burnout
- 📉 3× productivity loss from presenteeism (working while unwell)
- 🔍 52% of stressed employees are actively job hunting

---

## 🎯 Project Goal

Build an end-to-end ML pipeline that:
- Takes employee tech survey responses as input
- Analyzes behavioral and workplace patterns using ML
- Predicts whether an employee may need mental health support or treatment

---

## 📊 Dataset

- **Source:** [Mental Health in Tech Survey — Kaggle](https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey)
- **Rows:** 1,200+
- **Columns:** 27

**Key features used:**
`family_history`, `benefits`, `age`, `work_interfere`, `treatment`, `physical_health_consequence`, `coworkers`

---

## ⚙️ ML Pipeline

### Data Preprocessing
- Initial data examination
- Duplicate handling
- Null value handling
- Removal of irrelevant columns

### Training Steps
1. Target and feature selection (`treatment` as target)
2. Train-test split (80/20)
3. One-hot encoding of categorical columns
4. Normalization using `MinMaxScaler`
5. Model selection — `KNeighborsClassifier`
6. Model fit on training data
7. Evaluation on preprocessed test data

---

## 🤖 Model Comparison

| Model | Accuracy / R² Score |
|---|---|
| KNN | 0.7738 |
| Linear Regression | 0.4763 |
| Decision Tree Classifier | 0.7698 |
| **Random Forest Classifier** | **0.7976** ✅ |

### Ensemble Methods

| Model | R² Score |
|---|---|
| Bagging Classifier | 0.7778 |
| AdaBoost Classifier | 0.7698 |
| **Random Forest Classifier** | **0.8095** ✅ |
| Gradient Boosting Classifier | 0.7579 |

> Grid Search and Random Search were performed for hyperparameter tuning, but performance remained close to the default configuration — suggesting the dataset size is the primary limiting factor.

**Best model: Random Forest Classifier (R² = 0.8095)**

---

## 🚀 Deployment

The model is deployed as an interactive web app using **Streamlit**.

**Deployment architecture:**

```
Main Notebook                   Streamlit App
─────────────────               ─────────────────────────────
Save encoder   ──┐              Load encoder pickle
Save normalizer──┼──────────►  Load normalizer pickle
Save model     ──┘              Load model pickle
                │               Import helper functions
                ▼
          py file (functions)
          ─────────────────
          One-hot encoder fn
          Normalizer fn
```

🔗 **Live App:** [mentalhealthmlproject](https://mentalhealthmlproject.streamlit.app)

---

## 🗂️ Project Structure

```
mental-health-ml/
│
├── data/
│   └── survey.csv
│
├── notebooks/
│   └── mental_health_ml.ipynb     # Main training notebook
│
├── functions.py                   # Encoding & normalization helpers
├── app.py                         # Streamlit deployment file
│
├── models/
│   ├── encoder.pkl
│   ├── normalizer.pkl
│   └── model.pkl
│
└── README.md
```

---

## 💡 Key Learnings

- Model performance depends heavily on **data cleaning and encoding**, not just algorithm choice
- **Model choice affects interpretability** — tree-based models are easier to explain to stakeholders
- Human-centered data is complex and non-linear — simple rules rarely apply

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (KNN, Random Forest, Gradient Boosting, etc.)
- Streamlit
- Pickle

---

## 👩‍💻 Author

**Supriya Gir** — ML Bootcamp Project, 2026

