import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("StudentsPerformance.csv")

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("math score", axis=1)
y = df_encoded["math score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_train, y_train)

y_pred = ridge_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Ridge Regression MSE:", round(mse,2))
print("Ridge Regression R^2 Score:", round(r2,2))

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([0,100],[0,100], color='red', linestyle='--')
plt.xlabel("Actual Math Score")
plt.ylabel("Predicted Math Score")
plt.title("Ridge Regression: Actual vs Predicted Math Scores")
plt.grid(True)
plt.show()

coefficients = ridge_model.coef_
features = X.columns

plt.figure(figsize=(10,6))
plt.barh(features, coefficients, color='skyblue')
plt.title("Feature Importance (Ridge Coefficients)")
plt.xlabel("Coefficient Value")
plt.grid(True)
plt.tight_layout()
plt.show()