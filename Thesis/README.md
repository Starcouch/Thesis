# Machine Learning Analysis of Student Performance

### Author: Mathew Anand Prasad  
### Supervisor: Dr. Harangi Balázs  
### Institution: University of Debrecen
### Faculty: Faculty of Informatics
### Academic Year: 2025–2026  

---

## Overview

This project — **Machine Learning Analysis of Student Performance** — explores how **artificial intelligence** can support data-driven decision-making through **machine learning models** and **modern visualization techniques**.  

Using a real-world dataset of student performance, the system applies multiple regression algorithms to uncover key factors influencing students’ **math scores**, evaluates their performance, and visualizes insights to make the analytical process transparent and interpretable.

This project investigates factors influencing student math performance using supervised machine learning models. 
Rather than focusing solely on prediction accuracy, the study compares model behavior, feature importance, 
and prediction errors across demographic subgroups to support interpretability and fairness-aware analysis.

---

## Objectives

- Identify the most influential features affecting student math performance.
- Compare linear, regularized, and ensemble models in terms of predictive performance and interpretability.
- Evaluate model errors across demographic subgroups (e.g., gender, ethnicity, parental education).
- Apply cross-validation and hyperparameter tuning to ensure robust and reproducible results.  

---

## Dataset

The project uses the **Students Performance** dataset (`StudentsPerformance.csv`), which contains demographic and academic information about students, including:

| Feature | Description |
|----------|--------------|
| gender | Student gender (male/female) |
| race/ethnicity | Ethnic group of the student |
| parental level of education | Highest education level of the parents |
| lunch | Type of lunch (standard/free or reduced) |
| test preparation course | Completion status of a test preparation course |
| reading score | Student's reading score |
| writing score | Student's writing score |
| math score | Student's math score (target variable) |

**Goal:** Predict the **math score** based on all other features.

---

## Project Structure

### Final Thesis-Ready Pipeline
The following scripts form the core implementation of the thesis.
They represent the finalized, modular, and reproducible machine learning workflow used for analysis, evaluation, and interpretation.

| File                      | Description                                                                             |
| ------------------------- | --------------------------------------------------------------------------------------- |
| `preprocessing.py`        | Centralized data loading, train/test split, and preprocessing using `ColumnTransformer` |
| `train_models.py`         | Unified training of Linear, Ridge, Lasso, and Random Forest models using pipelines      |
| `tune_models.py`          | Hyperparameter tuning with 5-fold cross-validation (`GridSearchCV`)                     |
| `evaluate_models.py`      | Evaluation of tuned models using R² and MSE on the test set                             |
| `subgroup_analysis.py`    | Error analysis across demographic subgroups (gender, ethnicity, parental education)     |
| `interpretability.py`     | Model interpretability analysis (coefficients and feature importance comparison)        |
| `requirements.txt`        | List of dependencies required to reproduce the project                                  |
| `StudentsPerformance.csv` | Dataset used for analysis                                                               |
| `README.md`               | Project documentation                                                                   |

### Exploratory and Development Scripts
The repository also contains earlier exploratory and development scripts created during the initial stages of the project.
These files document the iterative development process but are not part of the final thesis pipeline.

| File                          | Description                                          |
| ----------------------------- | ---------------------------------------------------- |
| `01_load_data.py`             | Initial data loading and inspection                  |
| `02_visualize.py`             | Exploratory visualization of math score distribution |
| `03_group_analysis.py`        | Early group-based exploratory analysis               |
| `04_correlation.py`           | Correlation heatmap for feature relationships        |
| `05_predict_math_score.py`    | Baseline Linear Regression model                     |
| `06_visualize_predictions.py` | Visualization of predicted vs actual values          |
| `07_feature_importance.py`    | Feature importance from Linear Regression            |
| `08_random_forest_model.py`   | Standalone Random Forest implementation              |
| `09_ridge_regression.py`      | Standalone Ridge Regression implementation           |
| `10_lasso_regression.py`      | Standalone Lasso Regression implementation           |
| `11_compare_models.py`        | Initial model comparison script                      |
| `12_residual_analysis.py`     | Residual diagnostics for Lasso                       |
| `13_export_model.py`          | Early pipeline-based model export                    |
| `14_example_working_model.py` | Example inference using exported model               |
| `15_residual_analysis.py`     | Residual analysis for exported model                 |
| `16_final_model_export.py`    | Final standalone model export                        |

These scripts are retained for transparency and reproducibility, but the final analysis and results are based exclusively on the thesis-ready pipeline described above.

---

## Machine Learning Models Used
The study applies and compares multiple supervised regression models, selected to balance predictive performance, interpretability, and robustness.

| Model                       | Type               | Description                                                                                     |
| --------------------------- | ------------------ | ----------------------------------------------------------------------------------------------- |
| **Linear Regression**       | Baseline           | Serves as a reference model assuming linear relationships between features and math scores      |
| **Ridge Regression**        | Regularized Linear | Applies L2 regularization to reduce coefficient variance and mitigate multicollinearity         |
| **Lasso Regression**        | Regularized Linear | Applies L1 regularization to perform automatic feature selection and produce sparse models      |
| **Random Forest Regressor** | Ensemble           | Uses an ensemble of decision trees to capture non-linear relationships and feature interactions |

All models were implemented using a unified preprocessing and training pipeline to ensure fair and consistent comparison.

---

## Evaluation Methodology
Model performance was evaluated using a combination of cross-validation and held-out test set evaluation.

### Validation Strategy

- **5-fold cross-validation** was applied during model training using GridSearchCV to select optimal hyperparameters for Ridge, Lasso, and Random Forest models.
- **Cross-validation** ensured robust estimation of model performance and reduced sensitivity to data partitioning.

### Evaluation Metrics
The final tuned models were evaluated on an unseen test set using the following metrics:

- **R² Score (Coefficient of Determination)** Measures the proportion of variance in the math score explained by the model.
- **Mean Squared Error (MSE)** Measures the average squared difference between predicted and actual math scores, penalizing larger errors more strongly.

In addition to global performance metrics, error analysis across demographic subgroups was conducted to assess model behavior and potential performance disparities.

---

## Residual Analysis

Residuals (differences between predicted and actual values) were analyzed to:
- Assess overall model fit quality  
- Identify potential bias or non-linear patterns  
- Evaluate the distribution of prediction errors  

Residual diagnostics, including residual scatter plots, histograms, and Q–Q plots, were primarily conducted for the **Lasso Regression** model, which was selected as the final model due to its balance of predictive performance and interpretability.  

The residual analysis indicated no strong systematic patterns or severe deviations from normality, suggesting that the model provides stable and reliable predictions.

---

## Final Model

The final exported model (`final_model.pkl`) is a **Lasso Regression pipeline**, selected based on a balance of predictive performance, model simplicity, and interpretability.

The pipeline includes:
- **Data Preprocessing** — standardization of numerical features and one-hot encoding of categorical variables  
- **Feature Selection** — automatic feature selection via L1 regularization  
- **Prediction** — estimation of student math scores for new data samples  

### Example Usage

```python
import pickle
import pandas as pd

with open('final_model.pkl', 'rb') as f:
    model = pickle.load(f)

sample = pd.DataFrame({
    'gender': ['female'],
    'race/ethnicity': ['group B'],
    'parental level of education': ["bachelor's degree"],
    'lunch': ['standard'],
    'test preparation course': ['none'],
    'reading score': [85],
    'writing score': [90]
})

prediction = model.predict(sample)
print(f"Predicted Math Score: {prediction[0]:.2f}")
```
---

## Key Results

- **Selected Model:** Lasso Regression, chosen for its strong generalization performance and interpretability  
- **Predictive Performance:** Achieved competitive R² scores on the held-out test set with limited signs of overfitting  
- **Consistent Influential Features:**  
  - Reading score  
  - Writing score  
  - Test preparation course completion  
  - Parental education level  

These features were consistently identified as influential across multiple models, indicating their robust relationship with student math performance.

---

## Technologies and Tools

- **Programming Language:** Python 3  

- **Development Environment:**  
  - PyCharm (script-based development)

- **Libraries:**  
  - **Data Handling:** `pandas`, `numpy`  
  - **Machine Learning:** `scikit-learn` (pipelines, preprocessing, model training, and validation)  
  - **Visualization:** `matplotlib`, `seaborn`  
  - **Model Serialization:** `pickle`, `joblib`  

- **Key Techniques:**  
  - Scikit-learn Pipelines and `ColumnTransformer`  
  - One-hot encoding and feature standardization  
  - 5-fold cross-validation using `GridSearchCV`  

---

## Repository Contents
```
Thesis/
├── data/
│ └── StudentsPerformance.csv
│
├── src/
│ ├── preprocessing.py # Data loading, splitting, and preprocessing
│ ├── train_models.py # Unified model training using pipelines
│ ├── tune_models.py # Cross-validation and hyperparameter tuning
│ ├── evaluate_models.py # Model evaluation (R², MSE)
│ ├── subgroup_analysis.py # Error analysis across demographic groups
│ └── interpretability.py # Feature importance and model interpretability
│
├── models/
│ ├── final_model.pkl
│ ├── final_lasso_model.pkl
│ └── final_lasso_model_features.pkl
│
├── archive/
│ ├── 01_load_data.py
│ ├── 02_visualize.py
│ ├── 03_group_analysis.py
│ ├── 04_correlation.py
│ ├── 05_predict_math_score.py
│ ├── 06_visualize_predictions.py
│ ├── 07_feature_importance.py
│ ├── 08_random_forest_model.py
│ ├── 09_ridge_regression.py
│ ├── 10_lasso_regression.py
│ ├── 11_compare_models.py
│ ├── 12_residual_analysis.py
│ ├── 13_export_model.py
│ ├── 14_example_working_model.py
│ ├── 15_residual_analysis.py
│ └── 16_final_model_export.py
│
├── requirements.txt
└── README.md
```
---

## Conclusion

This project demonstrates the application of supervised machine learning methods to analyze factors influencing student math performance.  
By systematically comparing linear, regularized, and ensemble models, the study highlights the importance of model interpretability, robust validation, and error analysis in educational data analysis.

The results indicate that academic performance indicators such as reading and writing scores consistently play a central role in predicting math performance, while regularization techniques help balance predictive accuracy and model complexity.

---

## Future Work

- Extend the interpretability analysis using techniques such as SHAP values or partial dependence plots.  
- Integrate the trained model into an interactive Streamlit dashboard for exploratory analysis and demonstration purposes.  
- Investigate additional evaluation criteria, including fairness-aware metrics, to further assess subgroup performance.  
- Apply the proposed methodology to larger and more diverse educational datasets to evaluate generalizability.

---

## How to Run the Project
This section describes how to set up the environment and reproduce the experiments in this repository.

### Prerequisites

- Python 3.9+ (recommended)
- Git (for cloning the repository)
- A virtual environment (recommended)
- Development environment: PyCharm or any Python-compatible IDE

### Clone the Repository
```
git clone <https://github.com/Starcouch/Thesis>
cd Thesis
```
### Set Up a Virtual Environment (Recommended)
```
python -m venv .venv
```
Activate the environment:
- Windows:
```
.venv\Scripts\activate
```
- macOS / Linux:
```
source .venv/bin/activate
```
### Install Dependencies
All required dependencies are listed in requirements.txt.
```
pip install -r requirements.txt
```
### Project Structure
Ensure the repository follows the structure below:
```
Thesis
├── data/
│   └── StudentsPerformance.csv
├── src/
│   ├── preprocessing.py
│   ├── train_models.py
│   ├── tune_models.py
│   ├── evaluate_models.py
│   ├── subgroup_analysis.py
│   └── interpretability.py
├── models/
├── archive/
├── requirements.txt
└── README.md
```
### Running the Pipeline
All scripts should be executed from the project root directory.

- Train baseline models
```
python src/train_models.py
```
- Perform cross-validation and hyperparameter tuning
```
python src/tune_models.py
```
- Evaluate tuned models
```
python src/evaluate_models.py
```
- Perform subgroup error analysis
```
python src/subgroup_analysis.py
```
- Run interpretability analysis
```
python src/interpretability.py
```
### Model Artifacts
Trained models are stored in the models/ directory and can be loaded for inference or further analysis.

---

## Acknowledgements

Supervised by **Dr. Harangi Balázs**  
Special thanks to the **Department of Data Science and Visualization**, **Faculty of Informatics** for academic support and resources.