import pandas as pd
import numpy as np

df = pd.read_csv("data/Telco-Customer-Churn.csv")

# 1. The Critical Fix: Convert TotalCharges to numeric
# 'coerce' turns empty spaces into NaN so we can handle them
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# 2. Check for missing values created by the fix
print(f"Missing values in TotalCharges: {df['TotalCharges'].isnull().sum()}")

# 3. Drop rows with missing TotalCharges (only 11 rows usually)
df.dropna(inplace=True)


# Check Churn Distribution
print("--- Churn Counts ---")
print(df['Churn'].value_counts())

print("\n--- Churn Percentage ---")
print(df['Churn'].value_counts(normalize=True) * 100)


# 4. Remove CustomerID (it's useless for prediction)
df.drop('customerID', axis=1, inplace=True)

print("Data loaded and basic cleaning complete.")
print(df.head())