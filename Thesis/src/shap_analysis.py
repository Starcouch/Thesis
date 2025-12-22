import shap
import matplotlib.pyplot as plt
import pandas as pd

from tune_models import tune_models
from preprocessing import load_data, split_data


def run_shap_analysis():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    best_models = tune_models()
    rf_pipeline = best_models["RandomForest"]

    preprocessor = rf_pipeline.named_steps["preprocessor"]
    rf_model = rf_pipeline.named_steps["model"]

    X_train_transformed = preprocessor.transform(X_train)

    cat_features = preprocessor.transformers_[1][1].get_feature_names_out(
        preprocessor.transformers_[1][2]
    )
    feature_names = list(preprocessor.transformers_[0][2]) + list(cat_features)

    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_train_transformed)

    shap.summary_plot(
        shap_values,
        X_train_transformed,
        feature_names=feature_names,
        show=True
    )


if __name__ == "__main__":
    run_shap_analysis()
