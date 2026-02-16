import pickle

from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from preprocessing import load_data, split_data, build_preprocessor
from tune_models import tune_models


def export_final_model():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # Use tuned models instead of hardcoding
    best_models = tune_models()

    # Select the best model (Ridge in your case)
    final_pipeline = best_models["Ridge"]

    # Save the tuned pipeline directly
    with open("../models/final_model.pkl", "wb") as f:
        pickle.dump(final_pipeline, f)

    print("Final tuned Ridge model exported successfully.")


if __name__ == "__main__":
    export_final_model()
