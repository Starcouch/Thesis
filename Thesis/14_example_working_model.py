import pickle
import pandas as pd

with open('final_model.pkl', 'rb') as file:
    final_model = pickle.load(file)

sample_data = {
    'gender': ['female'],
    'race/ethnicity': ['group B'],
    'parental level of education': ['bachelor\'s degree'],
    'lunch': ['standard'],
    'test preparation course': ['none'],
    'reading score': [85],
    'writing score': [90]
}

sample_df = pd.DataFrame(sample_data)

predicted_score = final_model.predict(sample_df)
print(f"Predicted math score for sample: {predicted_score[0]:.2f}")
