import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def load_data(path="StudentsPerformance.csv"):
    return pd.read_csv(path)

def split_data(df, target="math score", test_size=0.2, random_state=42):
    X = df.drop(columns=[target])
    y = df[target]

    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state
    )


def build_preprocessor(X):
    numeric_features = ["reading score", "writing score"]
    categorical_features = [col for col in X.columns if col not in numeric_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    return preprocessor
