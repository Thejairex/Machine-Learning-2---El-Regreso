import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error

# from sklearn.ensemble import xgboost as xgb

class sellHouse:
    def __init__(self):
        self.train = pd.read_csv("train_normalized.csv")
        self.train.drop("Unnamed: 0", axis=1, inplace=True)   
        self.test = pd.read_csv("test_normalized.csv")
        self.test.drop("Unnamed: 0", axis=1, inplace=True)   
    
if __name__ == "__main__":
    sh = sellHouse()
    print(sh.train.head())