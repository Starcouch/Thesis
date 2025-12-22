import pandas as pd
from sklearn.metrics import mean_absolute_error

from preprocessing import load_data, split_data
from tune_models import tune_models


def subgroup_error_analysis(group_column):
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    best_models = tune_models()
    model = best_models["Lasso"]

    y_pred = model.predict(X_test)

    analysis_df = X_test.copy()
    analysis_df["y_true"] = y_test.values
    analysis_df["y_pred"] = y_pred
    analysis_df["abs_error"] = (analysis_df["y_true"] - analysis_df["y_pred"]).abs()

    group_errors = (
        analysis_df
        .groupby(group_column)["abs_error"]
        .mean()
        .sort_values()
    )

    print(f"\nMean Absolute Error by {group_column}:")
    print(group_errors)

    return group_errors


if __name__ == "__main__":
    subgroup_error_analysis("gender")
    subgroup_error_analysis("race/ethnicity")
    subgroup_error_analysis("parental level of education")
