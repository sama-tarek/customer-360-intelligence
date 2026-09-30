import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
palette = ['lightgreen', 'indianred']

df = pd.read_csv('../data/processed/customer_360_cleaned.csv')

df_features = df.copy()

print(f'Min months : {df_features['TenureMonths'].min()}')
print(f'Max months : {df_features['TenureMonths'].max()}')

bins = [1, 12, 24, 36, 48, 60, 72]
labels = ['<= Year', '2 Years', '3 Years', '4 Years', '5 Years', '6 Years']

df_features['TenureGroups'] = pd.cut(
    df_features['TenureMonths'], bins=bins, labels=labels)

df_features[['TenureMonths', 'TenureGroups']].head()

df_features['SupportCallsPerTenure'] = (
    df_features['SupportCalls6M'] / df_features['TenureMonths']).round(2)
df_features[['SupportCalls6M', 'TenureMonths',
             'SupportCallsPerTenure']].head(3)

df_features["LatePaymentRate"] = (df_features["LatePayments12M"] / 12) * 100
df_features[['LatePayments12M', 'LatePaymentRate']].tail(5)

df_features["UsagePerService"] = (
    df_features["MonthlyUsageGB"] / df_features["NumServices"]).round(2)
df_features[['MonthlyUsageGB', 'NumServices', 'UsagePerService']].head(3)

df_features["ChargePerService"] = (
    df_features["MonthlyChargeEGP"] / df_features["NumServices"]).round(2)
df_features[['MonthlyChargeEGP', 'NumServices', 'ChargePerService']].head(3)

df_features["EstimatedAnnualCharge"] = df_features["MonthlyChargeEGP"] * 12
df_features[['MonthlyChargeEGP', 'EstimatedAnnualCharge']].head(3)

df_features["EstimatedTotalCharge"] = df_features["MonthlyChargeEGP"] * \
    df_features['TenureMonths']
df_features[['MonthlyChargeEGP', 'TenureMonths',
             'EstimatedTotalCharge']].head(3)

threshold = df_features['EstimatedAnnualCharge'].median()

df_features['HighValueCustomer'] = (
    df_features['EstimatedAnnualCharge'] > threshold).astype(int)

print(f'Threshold: {threshold.round(2)}')
df_features[['EstimatedAnnualCharge', 'HighValueCustomer']].head(3)

df_features['SupportSatisfactionInteraction'] = (
    df_features['SupportCalls6M'] * (5 - df_features['SatisfactionScore']))

df_features[['SupportCalls6M', 'SatisfactionScore',
             'SupportSatisfactionInteraction']].head(3)

df_features.to_csv("../data/processed/customer_360_features.csv", index=False)
