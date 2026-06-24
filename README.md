# Student Placement Prediction — MLP Project

## Overview
An end-to-end Machine Learning project that predicts student campus placement outcomes  
using a Multi-Layer Perceptron (MLP) neural network.

- **Dataset**: Student Placement Prediction Dataset 2026 (Kaggle, 100,000 records, 26 features)
- **Model**: scikit-learn MLPClassifier
- **Interface**: Streamlit web application

---

## Folder Structure

```
internship_success_prediction/
├── dataset/
│   └── student_placement_prediction_dataset_2026.csv   ← Place Kaggle CSV here
├── notebooks/
│   └── model_training.ipynb                            ← Main training notebook
├── models/
│   ├── mlp_model.pkl
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   └── feature_names.pkl
├── results/
│   ├── eda_plots.png
│   ├── confusion_matrix.png
│   ├── accuracy.png
│   ├── feature_importance.png
│   └── classification_report.txt
├── app/
│   └── app.py                                          ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Download the dataset

Download from Kaggle:  
👉 https://www.kaggle.com/datasets/sehaj1104/student-placement-prediction-dataset-2026

Place the CSV at:
```
dataset/student_placement_prediction_dataset_2026.csv
```

### 3. Run the notebook

Open and run all cells in:
```
notebooks/model_training.ipynb
```

This will:
- Load and explore the dataset
- Preprocess (encode, scale, split)
- Train 3 MLP architectures and compare
- Generate all result plots to `results/`
- Save the best model to `models/`

### 4. Launch the web app

```bash
streamlit run app/app.py
```

---

## Model Details

| Architecture    | Hidden Layers   |
|-----------------|-----------------|
| MLP-1           | (32,)           |
| MLP-2 (baseline)| (64, 32)        |
| MLP-3           | (128, 64, 32)   |

**Target variable**: `placement_status` (Placed / Not Placed)

**Features used** (23 total):
- Academic: `age`, `cgpa`, `branch`, `college_tier`, `attendance_percentage`, `backlogs`, `study_hours_per_day`
- Technical: `coding_skill_score`, `aptitude_score`, `logical_reasoning_score`, `certifications_count`, `projects_count`, `github_repos`, `internships_count`
- Professional: `communication_skill_score`, `mock_interview_score`, `linkedin_connections`
- Personal: `extracurricular_score`, `leadership_score`, `volunteer_experience`, `sleep_hours`, `gender`

**Dropped** (leakage/irrelevant):
- `student_id` — unique identifier, not predictive
- `salary_package_lpa` — only available after placement (data leakage)

---

## Output Files

| File | Description |
|------|-------------|
| `results/eda_plots.png` | Placement distribution & CGPA histogram |
| `results/confusion_matrix.png` | Confusion matrix heatmap |
| `results/accuracy.png` | Architecture comparison bar chart |
| `results/feature_importance.png` | Permutation importance chart |
| `results/classification_report.txt` | Full metrics report |
# Internship-Placement-Prediction-MLP
