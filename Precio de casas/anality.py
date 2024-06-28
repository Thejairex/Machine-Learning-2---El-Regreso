import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split


def find_alpha(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Cargar el conjunto de datos (utilizando un dataset de ejemplo)


    # Crear un pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', Lasso())
    ])

    # Definir el rango de valores de alpha
    param_grid = {
        'model__alpha': [0.001, 0.01, 0.1, 1, 10, 100]
    }

    # Configurar y ejecutar GridSearchCV para Lasso
    grid_search_lasso = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search_lasso.fit(X_train, y_train)

    best_lasso = grid_search_lasso.best_estimator_
    print(f"Best Lasso alpha: {grid_search_lasso.best_params_['model__alpha']}")

    # Configurar y ejecutar GridSearchCV para Ridge
    pipeline.set_params(model=Ridge())
    grid_search_ridge = GridSearchCV(pipeline, param_grid, cv=5, scoring='neg_mean_squared_error')
    grid_search_ridge.fit(X_train, y_train)

    best_ridge = grid_search_ridge.best_estimator_
    print(f"Best Ridge alpha: {grid_search_ridge.best_params_['model__alpha']}")

    # Evaluar el mejor modelo en el conjunto de prueba
    lasso_score = best_lasso.score(X_test, y_test)
    ridge_score = best_ridge.score(X_test, y_test)

    print(f"Lasso score: {lasso_score}")
    print(f"Ridge score: {ridge_score}")
     
     
def correlation(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    alpha_lasso = 100.0
    alpha_rigde = 100.0
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    
    lasso = Lasso(alpha=alpha_lasso)
    ridge = Ridge(alpha=alpha_rigde)
    
    lasso.fit(X_train_scaled, y_train)
    ridge.fit(X_train_scaled, y_train)
    
    
    lasso_coefficients = lasso.coef_
    ridge_coefficients = ridge.coef_

    # Crear un DataFrame para visualizar las correlaciones
    coefficients_df = pd.DataFrame({
        'Feature': X.columns,
        'Lasso Coefficients': lasso_coefficients,
        'Ridge Coefficients': ridge_coefficients
    })

    lasso_sorted = coefficients_df.reindex(coefficients_df['Lasso Coefficients'].abs().sort_values(ascending=False).index)

    # Ordenar por la magnitud de los coeficientes de Ridge
    ridge_sorted = coefficients_df.reindex(coefficients_df['Ridge Coefficients'].abs().sort_values(ascending=False).index)

    print("Lasso - Características más importantes:")
    print(lasso_sorted.head(20))  # Mostrar las 10 características más importantes

    print("\nRidge - Características más importantes:")
    print(ridge_sorted.head(20))
    
    # print(coefficients_df)
     
     

if __name__ == "__main__":
    
    data = pd.read_csv("train_normalized.csv")
    X = data.drop('SalePrice', axis=1)
    y = data["SalePrice"]
    correlation(X, y)
