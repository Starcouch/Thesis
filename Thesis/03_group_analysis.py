import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("StudentsPerformance.csv")

sns.set(style="whitegrid")

plt.figure(figsize=(8,5))
sns.boxplot(x="test preparation course", y="math score", hue="test preparation course" , data=df, palette="Set2", legend=False)
plt.title("Math Score vs Test Preparation Course")
plt.xlabel("Test Preparation Course")
plt.ylabel("Math Score")
plt.show()

plt.figure(figsize=(10, 5))
sns.boxplot(x="parental level of education", y="math score", data=df, palette="pastel")
plt.title("Math Score vs Parental Education")
plt.xticks(rotation=45)
plt.xlabel("Parental Level of Education")
plt.ylabel("Math Score")
plt.tight_layout()
plt.show()
