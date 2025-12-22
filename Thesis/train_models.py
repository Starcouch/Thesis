from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from preprocessing import load_data, split_data, build_preprocessor


def train_models():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)

    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        results[name] = {
            "model": pipeline,
            "r2": r2_score(y_test, y_pred),
            "mse": mean_squared_error(y_test, y_pred)
        }

        print(f"{name}: R²={results[name]['r2']:.3f}, MSE={results[name]['mse']:.2f}")

    return results


if __name__ == "__main__":
    train_models()
