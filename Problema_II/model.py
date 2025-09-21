"""
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# --------------------------
# Implementación propia
# --------------------------

def linear_regression_custom(X_train, y_train):
    # TODO: Implementar regresión lineal manual
    pass

def random_forest_custom(X_train, y_train):
    # TODO: Implementar Random Forest manual (puede ser simplificado)
    pass

# --------------------------
# Implementación con sklearn
# --------------------------

def linear_regression_sklearn(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def random_forest_sklearn(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model

# --------------------------
# Predicción
# --------------------------

def predecir(model, X_test):
    return model.predict(X_test)

"""


