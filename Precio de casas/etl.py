import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder

def load_data(filename: str):
    return pd.read_csv(filename)


def check_missing_values(df):
    # comprobar la cantidad de valores faltantes
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0]
    print(missing.info)


def check_missing_columns(df):
    # Comprobar la distribución de los valores faltantes
    for column in df.columns:
        if np.dtypes.ObjectDType == type(df[column].dtype):
            if df[column].isnull().sum() > 0:
                print(df[column].value_counts())
                print("Vacios: ", df[column].isnull().sum())


def normalize(df):

    # Eliminar columnas que tengan el 30% de valores faltantes
    df.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence',
            'FireplaceQu', 'MasVnrType', 'Utilities'], axis=1, inplace=True)

    # Imputar valores
    imputer = SimpleImputer(strategy="most_frequent")
    df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Codificar categoricas nominales
    category_orders = {
        'MSZoning': ['C (all)', 'RH', 'FV', 'RM', 'RL'],
        'LandSlope': ['Sev', 'Mod', 'Gtl'],
        'ExterQual': ['Fa', 'TA', 'Gd', 'Ex'],
        'ExterCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'BsmtQual': ['Fa', 'TA', 'Gd', 'Ex'],
        'BsmtCond': ['Po', 'Fa', 'TA', 'Gd'],
        'BsmtFinType1': ['LwQ', 'Rec', 'BLQ', 'ALQ', 'GLQ', 'Unf'],
        'BsmtFinType2': ['GLQ', 'ALQ', 'BLQ', 'LwQ', 'Rec', 'Unf'],
        'HeatingQC': ['Po', 'Fa', 'Gd', 'TA', 'Ex'],
        'CentralAir': ['N', 'Y'],
        'KitchenQual': ['Fa', 'TA', 'Gd', 'Ex'],
        'Functional': ['Sev', 'Maj2', 'Maj1', 'Mod', 'Min1', 'Min2', 'Typ'],
        'GarageFinish': ['Fin', 'RFn', 'Unf'],
        'GarageQual': ['Po', 'Fa', 'TA', 'Gd', 'Ex'],
        'GarageCond': ['Po', 'Fa', 'TA', 'Gd', 'Ex']
    }
    df = pd.get_dummies(df, columns=["Street", "LotShape", "LandContour", "LotConfig", "Neighborhood", "Condition1", "Condition2", "BldgType", "HouseStyle", "RoofStyle",
                        "RoofMatl", "Exterior1st", "Exterior2nd", "Foundation", "BsmtExposure", "Heating", "Electrical", "GarageType", "PavedDrive", "SaleType", "SaleCondition"], drop_first=True)

    ordinal_columns = category_orders.keys()
    rest_df = df.drop(ordinal_columns, axis=1)
    encoder = OrdinalEncoder(categories=[
                             category_orders[col] for col in df.columns if col in category_orders.keys()])

    df_ordinal_encoded = pd.DataFrame(encoder.fit_transform(
        df[ordinal_columns]), columns=ordinal_columns)
    return pd.concat([df_ordinal_encoded, rest_df], axis=1)


if __name__ == "__main__":
    train_df = load_data("train.csv")
    test_df = load_data("test.csv")
    train_df = normalize(train_df)
    test_df = normalize(test_df)
    train_df.to_csv("train_normalized.csv")
    test_df.to_csv("test_normalized.csv")