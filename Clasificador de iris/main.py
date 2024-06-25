import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def linear_regression():
    x = iris.drop(["Species"], axis=1) 
    y = iris["Species"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    lg = LogisticRegression()
    lg.fit(x_train, y_train)
    y_pred = lg.predict(x_test)
    print('Precisión Regresión Logística: {}'.format(lg.score(x_train, y_train)))
    # 0.9666666666666667


def svm():
    x = iris.drop(["Species"], axis=1)
    y = iris["Species"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    svc = SVC()
    svc.fit(x_train, y_train)
    y_pred = svc.predict(x_test)
    print('Precisión SVC: {}'.format(svc.score(x_train, y_train)))
    #  0.9583333333333334

def knn():
    x = iris.drop(["Species"], axis=1)
    y = iris["Species"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    knn = KNeighborsClassifier()
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    print('Precisión KNN: {}'.format(knn.score(x_train, y_train)))
    # 0.95
    
def decision_tree():
    x = iris.drop(["Species"], axis=1)
    y = iris["Species"]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
    
    arbol = DecisionTreeClassifier()
    arbol.fit(x_train, y_train)
    y_pred = arbol.predict(x_test)
    print('Precisión Decision Tree: {}'.format(arbol.score(x_train, y_train)))
    # 1.0
    
if __name__ == "__main__":
    iris = pd.read_csv("iris.csv")
    iris.drop(["Id"], axis=1, inplace=True)
    
    decision_tree()

