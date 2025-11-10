# AI-Driven Data Analysis and Visualization

### Author: Mathew Anand Prasad  
### Supervisor: Dr. Harangi Balázs  
### Institution: University of Debrecen
### Faculty: Faculty of Informatics
### Academic Year: 2025–2026  

---

## Overview

This project — **AI-Driven Data Analysis and Visualization** — explores how **artificial intelligence** can support data-driven decision-making through **machine learning models** and **modern visualization techniques**.  

Using a real-world dataset of student performance, the system applies multiple regression algorithms to uncover key factors influencing students’ **math scores**, evaluates their performance, and visualizes insights to make the analytical process transparent and interpretable.

---

## Objectives

- To analyze and visualize relationships within real-world data using AI-driven techniques.  
- To develop and compare different regression-based machine learning models for predictive accuracy.  
- To apply data visualization methods that make insights clear and actionable.  
- To demonstrate how AI can enhance data analysis, interpretation, and transparency.  

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

The project follows a clear data science workflow, implemented through modular Python scripts:

| Step | File | Description |
|------|------|-------------|
| 01 | `01_load_data.py` | Loads dataset and displays data summary |
| 02 | `02_visualize.py` | Visualizes math score distribution |
| 03 | `03_group_analysis.py` | Compares math scores across categorical variables |
| 04 | `04_correlation.py` | Displays feature correlations using a heatmap |
| 05 | `05_predict_math_score.py` | Builds baseline Linear Regression model |
| 06 | `06_visualize_predictions.py` | Visualizes actual vs predicted math scores |
| 07 | `07_feature_importance.py` | Shows feature importance from Linear Regression |
| 08 | `08_random_forest_model.py` | Trains and evaluates a Random Forest model |
| 09 | `09_ridge_regression.py` | Builds Ridge Regression model |
| 10 | `10_lasso_regression.py` | Builds and saves Lasso model (`final_lasso_model.pkl`) |
| 11 | `11_compare_models.py` | Compares Linear, Ridge, Lasso, and Random Forest models |
| 12 | `12_residual_analysis.py` | Performs detailed residual analysis on Lasso |
| 13 | `13_export_model.py` | Builds preprocessing + model pipeline and saves it (`final_model.pkl`) |
| 14 | `14_example_working_model.py` | Loads final model and predicts math score for new data |
| 15 | `15_residual_analysis.py` | Residual analysis for exported model |
| 16 | `16_final_model_export.py` | Trains final standardized Lasso model and saves it |

---

## Machine Learning Models Used

| Model | Type | Description |
|--------|------|-------------|
| **Linear Regression** | Baseline | Predicts math scores using linear relationships between variables |
| **Ridge Regression** | Regularized Linear | Adds L2 penalty to reduce overfitting |
| **Lasso Regression** | Regularized Linear | Adds L1 penalty for feature selection |
| **Random Forest Regressor** | Ensemble | Combines decision trees to handle nonlinearity and feature interactions |

---

## Evaluation Metrics

Each model was evaluated using:
- **R² Score (Coefficient of Determination)** — measures how much variance in the target variable is explained by the model.  
- **Mean Squared Error (MSE)** — measures average squared difference between actual and predicted values.  

Both metrics were used to compare performance and select the final model.

---

## Residual Analysis

Residuals (errors between predicted and actual values) were analyzed to:
- Verify model fit quality  
- Check for bias or non-linearity  
- Confirm normal distribution of errors  

Residual plots, histograms, and Q-Q plots confirmed that the **Lasso Regression** model provided stable and unbiased predictions.

---

## Final Model

The final exported model (`final_model.pkl`) is a **Lasso Regression pipeline** that includes:
- **Data Preprocessing** — scaling of numerical features and one-hot encoding for categorical features  
- **Feature Selection** — via L1 regularization  
- **Prediction** — math score estimation for new data samples  

Example usage:
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

## Key Results

- **Best Model:** Lasso Regression  
- **R² Score:** High accuracy with minimal overfitting  
- **Most Influential Features:**  
  - Reading score  
  - Writing score  
  - Test preparation course completion  
  - Parental education level  

---

## Technologies and Tools

- **Programming:** Python 3  
- **Libraries:**  
  - Data handling: `pandas`, `numpy`  
  - Machine Learning: `scikit-learn`  
  - Visualization: `matplotlib`, `seaborn`  
  - Serialization: `pickle`, `joblib`  
- **Environment:** Jupyter Notebook / Python Scripts  

---

## Repository Contents
```
├── StudentsPerformance.csv
├── 01_load_data.py
├── 02_visualize.py
├── 03_group_analysis.py
├── 04_correlation.py
├── 05_predict_math_score.py
├── 06_visualize_predictions.py
├── 07_feature_importance.py
├── 08_random_forest_model.py
├── 09_ridge_regression.py
├── 10_lasso_regression.py
├── 11_compare_models.py
├── 12_residual_analysis.py
├── 13_export_model.py
├── 14_example_working_model.py
├── 15_residual_analysis.py
├── 16_final_model_export.py
├── final_model.pkl
├── final_lasso_model.pkl
├── final_lasso_model_features.pkl
└── README.md
```
---

## Conclusion

This thesis project demonstrates the application of **AI and machine learning** in analyzing and visualizing educational data.  
By building and evaluating multiple predictive models, it highlights the **interpretability and effectiveness of AI** in supporting **data-driven insights and decision-making**.

---

## Future Work

- Integrate the trained model into an interactive **Streamlit dashboard** for live visualization.  
- Explore **deep learning models** or **ensemble stacking** for improved prediction accuracy.  
- Apply similar methodology to **larger, real-world educational datasets**.

---

## Acknowledgements

Supervised by **Dr. Harangi Balázs**  
Special thanks to the **Department of Data Science and Visualization**, **Faculty of Informatics** for academic support and resources.
