# Machine Learning Analysis of Student Performance

### Author: Mathew Anand Prasad  
### Supervisor: Dr. Harangi Balázs  
### Institution: University of Debrecen  
### Faculty: Faculty of Informatics  
### Academic Year: 2025–2026  

---

## Overview

This project — **Machine Learning Analysis of Student Performance** — explores how artificial intelligence can support data-driven decision-making through supervised machine learning models and modern visualization techniques.

Using a real-world dataset of student performance, the system applies multiple regression algorithms to uncover key factors influencing students’ math scores, evaluates predictive performance, and visualizes insights to make the analytical process transparent and interpretable.

Rather than focusing solely on prediction accuracy, the study systematically compares model behavior, feature importance, residual patterns, and subgroup prediction errors to support interpretability and fairness-aware analysis.

A deployed demonstration of the final trained model is available at:

https://starcouch-thesis-student-performance-predictor.streamlit.app/

---

## Objectives

- Identify the most influential features affecting student math performance.
- Compare linear, regularized, bagging, and boosting models.
- Apply cross-validation and hyperparameter tuning.
- Conduct residual diagnostics for validation.
- Evaluate prediction errors across demographic subgroups.
- Deploy the final trained model using Streamlit.

---

## Dataset

The project uses the **Students Performance** dataset (`StudentsPerformance.csv`).

| Feature | Description |
|----------|-------------|
| gender | Student gender (male/female) |
| race/ethnicity | Ethnic group of the student |
| parental level of education | Highest education level of the parents |
| lunch | Type of lunch (standard/free or reduced) |
| test preparation course | Completion status of a test preparation course |
| reading score | Student's reading score |
| writing score | Student's writing score |
| math score | Student's math score (target variable) |

**Goal:** Predict the math score based on all other features.

---

## Project Structure

### Final Thesis-Ready Pipeline

The following scripts form the finalized, modular, and reproducible machine learning workflow.

| File | Description |
|------|------------|
| `preprocessing.py` | Data loading, train/test split, preprocessing using `ColumnTransformer` |
| `train_models.py` | Training of Linear, Ridge, Lasso, Random Forest, and XGBoost models |
| `tune_models.py` | Hyperparameter tuning using 5-fold cross-validation (`GridSearchCV`) |
| `evaluate_models.py` | Evaluation using R² and MSE |
| `subgroup_analysis.py` | Demographic subgroup error analysis |
| `interpretability.py` | Coefficient and feature importance analysis |
| `shap_analysis.py` | SHAP-based comparison for Random Forest and XGBoost |
| `export_final_model.py` | Exports the tuned final Lasso model |
| `requirements.txt` | Project dependencies |
| `StudentsPerformance.csv` | Dataset |
| `README.md` | Project documentation |

### Exploratory and Development Scripts

Earlier exploratory scripts are retained in the `archive/` directory for transparency and reproducibility.

---

## Machine Learning Models Used

| Model | Type | Description |
|-------|------|------------|
| Linear Regression | Baseline | Reference linear model |
| Ridge Regression | Regularized (L2) | Controls multicollinearity |
| Lasso Regression | Regularized (L1) | Embedded feature selection |
| Random Forest Regressor | Bagging Ensemble | Captures nonlinear interactions |
| XGBoost Regressor | Boosting Ensemble | Sequential residual minimization |

All models were implemented using a unified pipeline architecture.

---

## Evaluation Methodology

### Validation Strategy

- 5-fold cross-validation using `GridSearchCV`
- Held-out test set evaluation

### Evaluation Metrics

- **R² Score**
- **Mean Squared Error (MSE)**

Subgroup error analysis and SHAP-based interpretability were also conducted.

---

## Residual Analysis

Residual diagnostics were performed for the Lasso Regression model, including:

- Residual vs predicted plots  
- Residual histograms  
- Q–Q plots  

Ridge and Lasso demonstrated nearly identical predictive performance. Lasso was selected due to its embedded feature selection property, which improves interpretability and aligns with the residual diagnostics conducted during model validation.

No strong violations of model assumptions were observed.

---

## Final Model

The final exported model (`models/final_model.pkl`) is a tuned **Lasso Regression pipeline (alpha = 0.001)**.

The pipeline includes:

- Standardization of numerical features  
- One-hot encoding of categorical variables  
- L1 regularization for feature selection  

### Example Usage

```python
import pickle
import pandas as pd

with open("models/final_model.pkl", "rb") as f:
    model = pickle.load(f)

sample = pd.DataFrame({
    "gender": ["female"],
    "race/ethnicity": ["group B"],
    "parental level of education": ["bachelor's degree"],
    "lunch": ["standard"],
    "test preparation course": ["none"],
    "reading score": [85],
    "writing score": [90]
})

prediction = model.predict(sample)
print(f"Predicted Math Score: {prediction[0]:.2f}")
```
## Key Results

- Selected Model: Tuned Lasso Regression (alpha = 0.001)  
- Test Performance: R² ≈ 0.88 on the held-out test set  
- Regularized linear models outperformed ensemble models (Random Forest and XGBoost)  
- Ridge and Lasso demonstrated nearly identical predictive performance; Lasso was selected due to its embedded feature selection property and consistency with residual diagnostics  
- Most influential features:
  - Reading score  
  - Writing score  
  - Test preparation course completion  
  - Parental education level  

These features were consistently identified across multiple models, indicating robust and stable relationships with student math performance.

---

## Technologies and Tools

- Programming Language: Python 3  
- Development Environment: PyCharm  
- Deployment: Streamlit  

Libraries Used:

- pandas  
- numpy  
- scikit-learn  
- xgboost  
- shap  
- matplotlib  
- seaborn  
- pickle  

Key Techniques:

- Scikit-learn Pipelines  
- ColumnTransformer for preprocessing  
- One-hot encoding and feature standardization  
- 5-fold cross-validation using GridSearchCV  
- Residual diagnostics and subgroup error analysis  

---

## Repository Contents
```
Thesis/
├── app/
│   └── streamlit_app.py
│
├── archive/
│   ├── 01_load_data.py
│   ├── 02_visualize.py
│   ├── 03_group_analysis.py
│   ├── 04_correlation.py
│   ├── 05_predict_math_score.py
│   ├── 06_visualize_predictions.py
│   ├── 07_feature_importance.py
│   ├── 08_random_forest_model.py
│   ├── 09_ridge_regression.py
│   ├── 10_lasso_regression.py
│   ├── 11_compare_models.py
│   ├── 12_residual_analysis.py
│   ├── 13_export_model.py
│   ├── 14_example_working_model.py
│   ├── 15_residual_analysis.py
│   └── 16_final_model_export.py
│
├── data/
│   └── StudentsPerformance.csv
│
├── models/
│   ├── final_lasso_model.pkl
│   ├── final_lasso_model_features.pkl
│   └── final_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── tune_models.py
│   ├── evaluate_models.py
│   ├── subgroup_analysis.py
│   ├── interpretability.py
│   ├── shap_analysis.py
│   ├── export_final_model.py
│   ├── export_streamlit_model.py
│   └── requirements.txt
│
├── requirements.txt
└── README.md
```

---

## Conclusion

This project demonstrates the structured application of supervised machine learning methods to analyze student math performance. Through systematic model comparison, hyperparameter tuning, interpretability analysis, and residual diagnostics, the study highlights the importance of transparency and validation rigor in educational data analysis.

The results indicate that regularized linear models provide the strongest balance between predictive performance and interpretability for this dataset.

---

## Future Work

- Extend fairness-aware evaluation metrics  
- Apply the methodology to larger educational datasets  
- Explore neural network architectures for comparison  
- Enhance the Streamlit deployment with additional analytical features  

---

## How to Run the Project

This section describes how to set up the environment and reproduce the experiments in this repository.

### Prerequisites

- Python 3.9+ (recommended)
- Git
- A virtual environment (recommended)
- Any Python-compatible IDE (e.g., PyCharm, VS Code)

---

### Clone the Repository

```
git clone https://github.com/Starcouch/Thesis
cd Thesis
```

---

### Set Up a Virtual Environment (Recommended)

```
python -m venv .venv
```

Activate the environment:

**Windows:**
```
.venv\Scripts\activate
```

**macOS / Linux:**
```
source .venv/bin/activate
```

---

### Install Dependencies

```
pip install -r requirements.txt
```

---

### Project Structure

```
Thesis/
├── app/
│   └── streamlit_app.py
│
├── archive/
│   ├── 01_load_data.py
│   ├── ...
│   └── 16_final_model_export.py
│
├── data/
│   └── StudentsPerformance.csv
│
├── models/
│   ├── final_lasso_model.pkl
│   ├── final_lasso_model_features.pkl
│   └── final_model.pkl
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── tune_models.py
│   ├── evaluate_models.py
│   ├── subgroup_analysis.py
│   ├── interpretability.py
│   ├── shap_analysis.py
│   ├── export_final_model.py
│   ├── export_streamlit_model.py
│   └── requirements.txt
│
├── requirements.txt
└── README.md
```

---

### Running the Pipeline

All scripts should be executed from the project root directory.

**Train baseline models**
```
python src/train_models.py
```

**Perform cross-validation and hyperparameter tuning**
```
python src/tune_models.py
```

**Evaluate tuned models**
```
python src/evaluate_models.py
```

**Perform subgroup error analysis**
```
python src/subgroup_analysis.py
```

**Run coefficient-based interpretability analysis**
```
python src/interpretability.py
```

**Run SHAP-based interpretability analysis (Random Forest & XGBoost)**
```
python src/shap_analysis.py
```

**Export the final tuned Lasso model**
```
python src/export_final_model.py
```

**Run the Streamlit dashboard**
```
streamlit run app/streamlit_app.py
```
### Model Artifacts
Trained models are stored in the models/ directory and can be loaded for inference or further analysis.

---

## Acknowledgements

Supervised by **Dr. Harangi Balázs**  
Special thanks to the **Department of Data Science and Visualization**, **Faculty of Informatics** for academic support and resources.