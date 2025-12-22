import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv("StudentsPerformance.csv")  # adjust if your file name is different

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop("math score", axis=1)
y = df_encoded["math score"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

results = {
    "Model": [],
    "R2 Score": [],
    "MSE": []
}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results["Model"].append(name)
    results["R2 Score"].append(r2_score(y_test, y_pred))
    results["MSE"].append(mean_squared_error(y_test, y_pred))

results_df = pd.DataFrame(results)
print(results_df)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(results_df["Model"], results_df["R2 Score"], color="skyblue")
plt.title("Model Comparison - R² Score")
plt.ylabel("R² Score")
plt.xticks(rotation=15)

plt.subplot(1, 2, 2)
plt.bar(results_df["Model"], results_df["MSE"], color="salmon")
plt.title("Model Comparison - Mean Squared Error")
plt.ylabel("MSE")
plt.xticks(rotation=15)

plt.tight_layout()
plt.show()