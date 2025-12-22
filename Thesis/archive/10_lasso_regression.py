import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("StudentsPerformance.csv")
df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("math score", axis=1)
y = df_encoded["math score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)

y_pred = lasso.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Lasso Regression MSE: {mse:.2f}")
print(f"Lasso Regression R^2 Score: {r2:.2f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='orange')
plt.plot([0, 100], [0, 100], linestyle='--', color='red')
plt.title("Lasso Regression: Actual vs Predicted Math Scores")
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
coefficients = pd.Series(lasso.coef_, index=X.columns)
coefficients.sort_values().plot(kind='barh', color='lightcoral')
plt.title("Feature Importance (Lasso Coefficients)")
plt.xlabel("Coefficient Value")
plt.tight_layout()
plt.show()

import joblib

joblib.dump(lasso, 'final_lasso_model.pkl')

joblib.dump(X.columns.tolist(), 'final_lasso_model_features.pkl')
