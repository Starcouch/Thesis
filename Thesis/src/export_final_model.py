import pickle
from tune_models import tune_models


def export_final_model():
    print("Loading tuned models...")
    best_models = tune_models()

    # Select tuned Lasso model
    final_pipeline = best_models["Lasso"]

    with open("../models/final_model.pkl", "wb") as f:
        pickle.dump(final_pipeline, f)

    print("Final tuned Lasso model (alpha=0.001) exported successfully.")
    print("Saved as: models/final_model.pkl")


if __name__ == "__main__":
    export_final_model()
