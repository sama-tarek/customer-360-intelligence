import pandas as pd

import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv('../data/raw/customer_360_ml_workshop.csv')

print(f'Shape : {df.shape}')
df.head()

df.columns.to_list()

df.info()

df.describe().T

for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")

unique_values_cols = ['Region', 'ContractType',
                      'InternetType', 'AutoPay', 'Churn']

for col in unique_values_cols:
    print(f'{df[col].value_counts()}')
    print()

churn_percentage = df['Churn'].value_counts(normalize=True) * 100
print(churn_percentage.round(2))
