import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.decomposition import PCA


class Etl:
    def __init__(self, filename):
        self.filename = filename
        self.load_data(self.filename)

    def load_data(self, filename: str) -> pd.DataFrame:
        """
        filename: name of the csv file, without extension
        """
        self.df = pd.read_csv(filename + ".csv")
        self.isSale = "SalePrice" in self.df.columns
        if self.isSale:
            self.sale = self.df["SalePrice"]
            self.df.drop(["SalePrice"], axis=1, inplace=True)

        return self.df

    def drop_columns(self):
        self.df.drop(['Id', 'PoolQC', 'MiscFeature', 'Alley', 'Fence',
                      'FireplaceQu', 'MasVnrType', 'Utilities'], axis=1, inplace=True)
        self.df.dropna(inplace=True)

    def check_missing_values(self):
        # comprobar la cantidad de valores faltantes

        missing = self.df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        return missing

    def check_dataset(self):
        missing = self.check_missing_values()

        print("Numero de valores faltantes: \n", missing.sum())
        print("Cantidad de Columnas: \n", len(self.df.columns))

        self.check_value_columns()

    def check_value_columns(self):
        # Comprobar la distribución de los valores
        for column in self.df.columns:
            if np.dtypes.ObjectDType == type(self.df[column].dtype):
                if self.df[column].isnull().sum() > 0:
                    print(self.df[column].value_counts())
                    print("Vacios: ", self.df[column].isnull().sum())

    def correlation(self):
        norm = (self.df - self.df.mean()) / self.df.std()
        pca = PCA(n_components=10)
        pca_result = pca.fit_transform(norm)

        pca_df = pd.DataFrame(pca_result, columns=[
                              "PCA "+str(i) for i in range(10)])

        print(pca_df.isna().sum().sum())
        correlation_matrix = pca_df.corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Matriz de Correlación')

        plt.show()

    def len_columns(self):
        print(len(self.df.columns))

    def save_normalized(self):
        if self.isSale:
            self.df["SalePrice"] = self.sale
            self.df.to_csv(self.filename + "_normalized.csv", index=False)
        else:
            self.df.to_csv(self.filename + "_normalized.csv", index=False)

    def normalize(self):

        if os.path.isfile(self.filename + "_normalized.csv"):
            self.df = self.load_data(self.filename + "_normalized.csv")

        else:
            category_orders = ["MSZoning", "LandSlope", "ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "BsmtFinType1",
                               "BsmtFinType2", "HeatingQC", "CentralAir", "KitchenQual", "Functional", "GarageFinish", "GarageQual", "GarageCond"]
            nominals = ["Street", "LotShape", "LandContour", "LotConfig", "Neighborhood", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st",
                        "Exterior2nd", "Foundation", "BsmtExposure", "Heating", "Electrical", "GarageType", "PavedDrive", "SaleType", "SaleCondition", "Condition1", "Condition2"]

            encoder_hot = OneHotEncoder(sparse_output=False, drop='first')
            onehot = encoder_hot.fit_transform(self.df[nominals])
            onehot_df = pd.DataFrame(onehot, columns=encoder_hot.get_feature_names_out(
                nominals), index=self.df.index)

            encoder_ord = OrdinalEncoder()
            ordinal = encoder_ord.fit_transform(self.df[category_orders])
            ordinal_df = pd.DataFrame(
                ordinal, columns=category_orders, index=self.df.index)

            self.df.drop(columns=nominals + category_orders, inplace=True)
            self.df = pd.concat([self.df, onehot_df, ordinal_df], axis=1)

            del category_orders, nominals, onehot, encoder_ord, ordinal

            self.save_normalized()


if __name__ == "__main__":
    train = Etl("train")
    train.drop_columns()
    train.normalize()
    train.check_dataset()
    print("-" * 40)
    test = Etl("test")
    test.drop_columns()
    test.normalize()
    test.check_dataset()
    # train.correlation()
