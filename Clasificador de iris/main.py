import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

class Iris:
    def __init__(self):
        iris = pd.read_csv("iris.csv")
        iris.drop(["Id"], axis=1, inplace=True)
        self.x = iris.drop(["Species"], axis=1) 
        self.y = iris["Species"]
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=0.2, random_state=0)

        
    def train_models(self, model):
        model.fit(self.x_train, self.y_train)
        
        return model
    
    
    def evaluate_model(self, model):
        y_pred = model.predict(self.x_test)
        print('Reporte de Clasificación:\n', classification_report(self.y_test, y_pred))
        print("Matriz de Confusión", confusion_matrix(self.y_test, y_pred))
        print('Precisión Entrenamiento: {}'.format(model.score(self.x_train, self.y_train)))
        print('Precisión Testeo: {}'.format(model.score(self.x_test, self.y_test)))
        print('Exactitud: ', accuracy_score(self.y_test, y_pred))
        print("Validacion cruzada: ", cross_val_score(model, self.x, self.y, cv=4))

    def graf_maxtrix(self, matrix):
        plt.figure(figsize=(10,7))
        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=self.y.unique(), yticklabels=self.y.unique())
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Matriz de Confusión')
        plt.show()
    

if __name__ == "__main__":
    iris = Iris()
    models = [LogisticRegression(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier()]
    
    for model in models:
        model = iris.train_models(model)
        iris.evaluate_model(model)
        print("-" * 40)