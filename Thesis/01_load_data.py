import pandas as pd

df = pd.read_csv("StudentsPerformance.csv")

print("HEAD OF DATA:")
print(df.head())

print("\nDATA INFO:")
print(df.info())

print("\nDESCRIPTIVE STATISTICS:")
print(df.describe())