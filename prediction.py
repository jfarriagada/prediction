#!/usr/bin/env python3
# encoding: utf-8

import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor

data = pd.read_csv("bateria.csv")
x = data["Tiempo"]
y = data["Carga"]
X = x[:,np.newaxis] 

while True:
    # Entrenamiento de los datos
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    # Predicción
    mlr = MLPRegressor(
        solver='lbfgs', 
        alpha=1e-5, 
        hidden_layer_sizes=(3,3),
        random_state=1
        )
    mlr.fit(X_train, y_train)

    minutes = 92
    print(mlr.score(X_train, y_train))

    if mlr.score(X_train, y_train) > 0.95:
        """
            Terminar la ejecución del programa cuando el modelo
            tenga un score > 0.95 
        """
        break

print("Predicción en %s minutos : " % minutes, mlr.predict(np.array([minutes]).reshape(1, 1)))