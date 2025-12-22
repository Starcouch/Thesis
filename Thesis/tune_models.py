from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

from preprocessing import load_data, split_data, build_preprocessor


def tune_models():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)

    models_and_params = {
        "Ridge": {
            "model": Ridge(),
            "params": {
                "model__alpha": [0.01, 0.1, 1, 10, 100]
            }
        },
        "Lasso": {
            "model": Lasso(max_iter=5000),
            "params": {
                "model__alpha": [0.001, 0.01, 0.1, 1]
            }
        },
        "RandomForest": {
            "model": RandomForestRegressor(random_state=42),
            "params": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 5, 10],
                "model__min_samples_split": [2, 5]
            }
        }
    }

    best_models = {}

    for name, config in models_and_params.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", config["model"])
            ]
        )

        grid = GridSearchCV(
            pipeline,
            config["params"],
            cv=5,
            scoring="r2",
            n_jobs=-1
        )

        grid.fit(X_train, y_train)

        best_models[name] = grid.best_estimator_

        print(f"\n{name}")
        print(f"Best CV R²: {grid.best_score_:.3f}")
        print(f"Best Params: {grid.best_params_}")

    return best_models


if __name__ == "__main__":
    tune_models()
