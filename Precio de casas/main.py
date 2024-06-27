import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder

df = pd.read_csv("train.csv")
missing = df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]
# print(missing.info)

# Eliminar columnas que tengan el 30% de valores faltantes
df.drop(['Id','PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType', 'Utilities'], axis=1, inplace=True)

missing = df.isna().sum().sort_values(ascending=False)
missing = missing[missing > 0]

df = pd.get_dummies(df, columns=["Street","LotShape","LandContour"])

print(df.head())
# for column in df.columns:
#     if np.dtypes.ObjectDType == type(df[column].dtype):
#         print(df[column].value_counts())
        
