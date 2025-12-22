import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("StudentsPerformance.csv")
X = data.drop("math score", axis=1)
y = data["math score"]

with open("final_lasso_model.pkl", "rb") as file:
    lasso_model = pickle.load(file)

y_pred = lasso_model.predict(X)

residuals = y - y_pred

plt.figure(figsize=(8, 5))
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Predicted Math Score")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True, bins=25, color='skyblue')
plt.axvline(0, color='red', linestyle='--')
plt.title("Distribution of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Lasso Regression MSE: {mse:.2f}")
print(f"Lasso Regression R² Score: {r2:.2f}")
