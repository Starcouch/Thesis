import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from tune_models import tune_models


def get_feature_names(preprocessor):
    num_features = preprocessor.transformers_[0][2]
    cat_features = (
        preprocessor
        .transformers_[1][1]
        .get_feature_names_out(preprocessor.transformers_[1][2])
    )
    return np.concatenate([num_features, cat_features])


def plot_linear_coefficients(model, model_name):
    preprocessor = model.named_steps["preprocessor"]
    regressor = model.named_steps["model"]

    feature_names = get_feature_names(preprocessor)
    coefs = regressor.coef_

    coef_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs,
        "AbsCoeff": abs(coefs)
    }).sort_values("AbsCoeff")

    plt.figure(figsize=(10, 8))
    plt.barh(coef_df["Feature"], coef_df["Coefficient"])
    plt.title(f"Feature Influence – {model_name}")
    plt.xlabel("Coefficient Value")
    plt.tight_layout()
    plt.show()


def plot_rf_importance(model):
    preprocessor = model.named_steps["preprocessor"]
    rf = model.named_steps["model"]

    feature_names = get_feature_names(preprocessor)
    importances = rf.feature_importances_

    imp_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    }).sort_values("Importance")

    plt.figure(figsize=(10, 8))
    plt.barh(imp_df["Feature"], imp_df["Importance"])
    plt.title("Feature Importance – Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()


def run_interpretability():
    best_models = tune_models()

    plot_linear_coefficients(best_models["Ridge"], "Ridge Regression")
    plot_linear_coefficients(best_models["Lasso"], "Lasso Regression")
    plot_rf_importance(best_models["RandomForest"])


if __name__ == "__main__":
    run_interpretability()
