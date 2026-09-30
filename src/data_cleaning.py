import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('../data/raw/customer_360_ml_workshop.csv')

df.duplicated().sum()

missing_count = (df.isnull().sum()).sort_values(ascending=False)

missing_percentage = (df.isnull().mean() * 100).sort_values(ascending=False)

missing_summary = pd.DataFrame({
    'Missing Count': missing_count[missing_count > 0],
    'Missing Percentage': missing_percentage[missing_percentage > 0]
})

missing_summary

df['MonthlyUsageGB'] = df['MonthlyUsageGB'].fillna(
    df['MonthlyUsageGB'].median())
df['SatisfactionScore'] = df['SatisfactionScore'].fillna(
    df['SatisfactionScore'].median())

df['InternetType'] = df['InternetType'].fillna(df['InternetType'].mode()[0])

print(f'Impossible Age Values: {len(df[df["Age"] < 0])}')
print(f'Impossible TenureMonths Values: {len(df[df["TenureMonths"] < 0])}')
print(
    f'Impossible MonthlyChargeEGP Values: {len(df[df["MonthlyChargeEGP"] < 0])}')
print(f'Impossible MonthlyUsageGB Values: {len(df[df["MonthlyUsageGB"] < 0])}')
print(f'Impossible NumServices Values: {len(df[df["NumServices"] < 0])}')
print(f'Impossible SupportCalls6M Values: {len(df[df["SupportCalls6M"] < 0])}')
print(
    f'Impossible LatePayments12M Values: {len(df[df["LatePayments12M"] < 0])}')

missing_summary_final = pd.DataFrame({
    "Missing Values": df.isnull().sum(),
    "Missing Percentage": (df.isnull().sum() / len(df)) * 100
})

missing_summary_final[
    missing_summary_final["Missing Values"] > 0
]

df.to_csv("../data/processed/customer_360_cleaned.csv", index=False)
