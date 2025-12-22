import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from preprocessing import load_data, split_data
from tune_models import tune_models


def evaluate_models():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    best_models = tune_models()

    results = []

    for name, model in best_models.items():
        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "R2": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)
    print("\nOverall Model Performance:")
    print(results_df)

    return best_models, X_test, y_test


if __name__ == "__main__":
    evaluate_models()
