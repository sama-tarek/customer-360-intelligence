import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
palette = ['lightgreen', 'indianred']

warnings.filterwarnings('ignore', category=FutureWarning)

df_cleaned = pd.read_csv('../data/processed/customer_360_cleaned.csv')

churn_counts = df_cleaned["Churn"].value_counts()
churn_pct = df_cleaned["Churn"].value_counts(normalize=True) * 100

churn_df = pd.DataFrame({
    "Count": churn_counts,
    "Percentage": churn_pct.round(2)
})

churn_df

categories = churn_df.index
values = churn_df['Count']

fig, ax = plt.subplots(1, 1, figsize=(10, 5))

bars = ax.bar(categories, values, color=palette)

for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            height, str(height), ha='center', va='bottom')

ax.spines['top'].set_color('None')
ax.spines['right'].set_color('None')

ax.set_xlabel('Churn')
ax.set_ylabel('Customers')
ax.set_title('Churn Distribution', fontweight='bold')

plt.tight_layout()
plt.savefig('../images/Churn Distribution.png')

plt.show()

churn_by_contract = pd.crosstab(
    df_cleaned["ContractType"],
    df_cleaned["Churn"],
    normalize="index"
) * 100

churn_by_contract

plt.figure(figsize=(10, 5))

sns.countplot(data=df_cleaned, x='ContractType',
              hue='Churn', hue_order=['No', 'Yes'], palette=palette)

plt.ylabel('Count')
plt.xlabel('Contract Type')
plt.title('Churn Rate by Contract Type')

plt.legend()
plt.tight_layout()

plt.savefig('../images/Churn by Contract Type.png')
plt.show()

churn_by_autopay = pd.crosstab(
    df_cleaned['AutoPay'],
    df_cleaned['Churn'],
    normalize='index'
) * 100

churn_by_autopay.round(2)

plt.figure(figsize=(10, 5))

sns.countplot(data=df_cleaned, x='AutoPay',
              hue='Churn', hue_order=['No', 'Yes'], palette=palette)

plt.ylabel('Count')
plt.xlabel('Auto Pay')
plt.title('Churn Rate by Auto Pay')

plt.legend()
plt.tight_layout()

plt.savefig('../images/Churn by Auto Pay.png')
plt.show()

churn_by_SatisfactionScore = pd.crosstab(
    df_cleaned['SatisfactionScore'],
    df_cleaned['Churn'],
    normalize='index'
) * 100

churn_by_SatisfactionScore.round(2)

plt.figure(figsize=(10, 5))

sns.countplot(data=df_cleaned, x='SatisfactionScore',
              hue='Churn', hue_order=['No', 'Yes'], palette=palette)

plt.xlabel('Satisfaction Score')
plt.title('Churn By Satisfaction Score', fontweight='bold')

plt.savefig('../images/Churn By Satisfaction Score.png')
plt.show()

numeric_cols = df_cleaned.select_dtypes(include=np.number).columns.tolist()

df_cleaned[numeric_cols].hist(figsize=(14, 10), bins=25, color='slategray')

plt.suptitle('Numerical Feature Distributions')

plt.tight_layout()

plt.savefig('../images/Numerical Distribution.png')
plt.show()
