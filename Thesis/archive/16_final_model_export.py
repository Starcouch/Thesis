import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

df = pd.read_csv('StudentsPerformance.csv')

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('math score', axis=1)
y = df_encoded['math score']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

final_model = Pipeline([
    ('scaler', StandardScaler()),
    ('lasso', Lasso(alpha=0.1))
])

final_model.fit(X_train, y_train)

with open('final_model.pkl', 'wb') as f:
    pickle.dump(final_model, f)

print("Final model trained and saved as 'final_model.pkl'")
