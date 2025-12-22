import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("StudentsPerformance.csv")

sns.set(style="whitegrid")

plt.figure(figsize=(8, 5))
sns.histplot(df["math score"], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Math Scores")
plt.xlabel("Math Score")
plt.ylabel("Number of Students")
plt.show()


