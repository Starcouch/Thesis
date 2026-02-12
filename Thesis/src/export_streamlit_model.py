import os
import pickle
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso

from preprocessing import load_data, split_data, build_preprocessor


def export_model():
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    preprocessor = build_preprocessor(X_train)

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", Lasso(alpha=0.1))
    ])

    pipeline.fit(X_train, y_train)

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "final_model.pkl")

    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)

    print("Streamlit-safe model exported successfully.")


if __name__ == "__main__":
    export_model()
