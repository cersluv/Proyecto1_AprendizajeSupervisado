"""
# model.py - Problema 3

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# Implementación propia
# --------------------------

def knn_custom(X_train, y_train, k=3):
    # TODO: Implementar KNN manual
    pass

def random_forest_custom(X_train, y_train):
    # TODO: Implementar Random Forest manual (simplificado)
    pass

# --------------------------
# Implementación con sklearn
# --------------------------

def knn_sklearn(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def random_forest_sklearn(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# --------------------------
# Predicción
# --------------------------

def predecir(model, X_test):
    return model.predict(X_test)
    
"""