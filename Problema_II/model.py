import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from collections import Counter

# --------------------------
# Implementación propia
# --------------------------

def linear_regression_custom(X_train, y_train):
    """
    Implementación manual de regresión lineal usando mínimos cuadrados ordinarios.
    Fórmula: θ = (X^T * X)^(-1) * X^T * y
    """
    # Convertir a numpy arrays si son DataFrames de pandas
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    
    # Agregar columna de unos para el término independiente (bias)
    X_with_bias = np.column_stack([np.ones(X_train.shape[0]), X_train])
    
    # Calcular los coeficientes usando la fórmula de mínimos cuadrados
    XtX = np.dot(X_with_bias.T, X_with_bias)
    Xty = np.dot(X_with_bias.T, y_train)
    
    # Resolver el sistema de ecuaciones
    try:
        coefficients = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:
        # Si la matriz no es invertible, usar pseudoinversa
        coefficients = np.linalg.pinv(XtX).dot(Xty)
    
    # Crear clase modelo personalizada
    class CustomLinearModel:
        def __init__(self, coefficients):
            self.intercept_ = coefficients[0]
            self.coef_ = coefficients[1:]
            self.coefficients = coefficients
        
        def predict(self, X):
            # Convertir a numpy array si es DataFrame
            if hasattr(X, 'values'):
                X = X.values
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            return np.dot(X_with_bias, self.coefficients)
    
    return CustomLinearModel(coefficients)


class DecisionTreeCustom:
    """Implementación simplificada de un árbol de decisión para regresión"""
    
    def __init__(self, max_depth=3, min_samples_split=10, min_samples_leaf=5, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.tree = None
    
    def fit(self, X, y):
        # Fijar semilla para reproducibilidad
        np.random.seed(self.random_state)
        
        # Convertir a numpy arrays si son DataFrames de pandas
        if hasattr(X, 'values'):
            X = X.values
        if hasattr(y, 'values'):
            y = y.values
        self.tree = self._build_tree(X, y, depth=0)
    
    def _build_tree(self, X, y, depth):
        n_samples, n_features = X.shape
        
        # Condiciones de parada más estrictas
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_samples < 2 * self.min_samples_leaf or
            len(np.unique(y)) == 1 or
            np.var(y) < 1e-6):  # Varianza muy pequeña
            return np.mean(y)
        
        # Encontrar la mejor división
        best_feature, best_threshold = self._find_best_split(X, y)
        
        if best_feature is None:
            return np.mean(y)
        
        # Dividir los datos
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Verificar que ambos grupos tengan suficientes muestras
        if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
            return np.mean(y)
        
        # Construir subárboles recursivamente
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def _find_best_split(self, X, y):
        best_mse = float('inf')
        best_feature = None
        best_threshold = None
        
        n_samples, n_features = X.shape
        
        # Limitar el número de características a probar (más eficiente)
        max_features = max(1, int(np.sqrt(n_features)))
        features_to_try = np.random.choice(n_features, size=min(max_features, n_features), replace=False)
        
        for feature in features_to_try:
            feature_values = X[:, feature]
            
            # Solo probar algunos umbrales (más eficiente)
            unique_values = np.unique(feature_values)
            if len(unique_values) <= 10:
                thresholds = unique_values
            else:
                # Tomar percentiles para reducir cálculo
                thresholds = np.percentile(unique_values, [10, 25, 50, 75, 90])
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Verificar que ambos grupos tengan suficientes muestras
                if (np.sum(left_mask) < self.min_samples_leaf or 
                    np.sum(right_mask) < self.min_samples_leaf):
                    continue
                
                # Calcular MSE después de la división
                left_var = np.var(y[left_mask]) if np.sum(left_mask) > 1 else 0
                right_var = np.var(y[right_mask]) if np.sum(right_mask) > 1 else 0
                
                weighted_mse = (left_var * np.sum(left_mask) + right_var * np.sum(right_mask)) / n_samples
                
                if weighted_mse < best_mse:
                    best_mse = weighted_mse
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def predict(self, X):
        # Convertir a numpy array si es DataFrame
        if hasattr(X, 'values'):
            X = X.values
        return np.array([self._predict_single(x, self.tree) for x in X])
    
    def _predict_single(self, x, tree):
        if isinstance(tree, (int, float)):
            return tree
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])


def random_forest_custom(X_train, y_train, n_estimators=100, max_depth=None, 
                        min_samples_split=2, min_samples_leaf=1, random_state=42):
    """
    Implementación simplificada de Random Forest.
    Ahora acepta los mismos parámetros que sklearn para comparación justa.
    """
    # Convertir a numpy arrays si son DataFrames de pandas
    if hasattr(X_train, 'values'):
        X_train = X_train.values
    if hasattr(y_train, 'values'):
        y_train = y_train.values
    
    # Fijar semilla principal
    np.random.seed(random_state)
    
    class CustomRandomForest:
        def __init__(self, trees):
            self.trees = trees
        
        def predict(self, X):
            # Convertir a numpy array si es DataFrame
            if hasattr(X, 'values'):
                X = X.values
            # Promedio de las predicciones de todos los árboles
            predictions = np.array([tree.predict(X) for tree in self.trees])
            return np.mean(predictions, axis=0)
    
    trees = []
    n_samples, n_features = X_train.shape
    
    print(f"Entrenando Random Forest personalizado con {n_estimators} árboles...")
    
    for i in range(n_estimators):
        if i % 10 == 0:  # Mostrar progreso cada 10 árboles
            print(f"Árbol {i+1}/{n_estimators}")
        
        # Bootstrap sampling (muestreo con reemplazo)
        # Usar una semilla diferente para cada árbol para variabilidad
        np.random.seed(random_state + i)
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        X_bootstrap = X_train[bootstrap_indices]
        y_bootstrap = y_train[bootstrap_indices]
        
        # Crear y entrenar árbol con los mismos parámetros que sklearn
        tree = DecisionTreeCustom(
            max_depth=max_depth if max_depth is not None else 10,  # Si es None, usar un valor razonable
            min_samples_split=min_samples_split, 
            min_samples_leaf=min_samples_leaf,
            random_state=random_state + i
        )
        tree.fit(X_bootstrap, y_bootstrap)
        trees.append(tree)
    
    print("✓ Random Forest personalizado entrenado")
    return CustomRandomForest(trees)


# --------------------------
# Implementación con sklearn + GridSearch
# --------------------------

def optimize_random_forest_sklearn(X_train, y_train, random_state=42):
    """
    Encuentra los mejores hiperparámetros para Random Forest usando GridSearchCV
    """
    print("Optimizando hiperparámetros de Random Forest con GridSearchCV...")
    
    # Definir grid de parámetros a probar
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Crear modelo base
    rf = RandomForestRegressor(random_state=random_state)
    
    # GridSearchCV
    grid_search = GridSearchCV(
        rf, 
        param_grid, 
        cv=5,  # 5-fold cross validation
        scoring='neg_mean_squared_error',
        n_jobs=-1,  # Usar todos los cores disponibles
        verbose=1
    )
    
    # Entrenar
    grid_search.fit(X_train, y_train)
    
    # Mostrar mejores parámetros
    print("\n=== MEJORES PARÁMETROS ENCONTRADOS ===")
    print(f"Mejores parámetros: {grid_search.best_params_}")
    print(f"Mejor score CV (MSE): {-grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_


def linear_regression_sklearn(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model


def random_forest_sklearn(X_train, y_train):
    """Versión básica sin optimización (para comparar)"""
    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    return model


# --------------------------
# Predicción
# --------------------------

def predecir(model, X_test):
    return model.predict(X_test)

# --------------------------
# Evaluación
# --------------------------

def evaluar_modelo(y_true, y_pred, nombre_modelo="Modelo"):
    print(f"\n--- Evaluación: {nombre_modelo} ---")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred):.4f}")
    print(f"MSE:  {mean_squared_error(y_true, y_pred):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.4f}")
    print(f"R²:   {r2_score(y_true, y_pred):.4f}")

def comparar_modelos(y_true, y_pred_custom, y_pred_sklearn, nombre_modelo="Modelo"):
    """Función para comparar implementación manual vs sklearn"""
    print(f"\n=== COMPARACIÓN: {nombre_modelo} ===")
    
    print("\n--- Implementación Manual ---")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred_custom):.4f}")
    print(f"MSE:  {mean_squared_error(y_true, y_pred_custom):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_custom)):.4f}")
    print(f"R²:   {r2_score(y_true, y_pred_custom):.4f}")
    
    print("\n--- Implementación sklearn ---")
    print(f"MAE:  {mean_absolute_error(y_true, y_pred_sklearn):.4f}")
    print(f"MSE:  {mean_squared_error(y_true, y_pred_sklearn):.4f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred_sklearn)):.4f}")
    print(f"R²:   {r2_score(y_true, y_pred_sklearn):.4f}")
    
    # Diferencia entre implementaciones
    mae_diff = abs(mean_absolute_error(y_true, y_pred_custom) - mean_absolute_error(y_true, y_pred_sklearn))
    print(f"\n--- Diferencia MAE entre implementaciones: {mae_diff:.4f} ---")

# --------------------------
# Visualización
# --------------------------

def graficar_resultados(y_true, y_pred, nombre_modelo="Modelo"):
    plt.figure(figsize=(8, 5))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel("Valor real")
    plt.ylabel("Predicción")
    plt.title(f"Predicción vs Real - {nombre_modelo}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def graficar_comparacion(y_true, y_pred_custom, y_pred_sklearn, nombre_modelo="Modelo"):
    """Función para graficar comparación entre implementaciones"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Implementación manual
    ax1.scatter(y_true, y_pred_custom, alpha=0.5, color='blue')
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax1.set_xlabel("Valor real")
    ax1.set_ylabel("Predicción")
    ax1.set_title(f"{nombre_modelo} - Implementación Manual")
    ax1.grid(True)
    
    # Implementación sklearn
    ax2.scatter(y_true, y_pred_sklearn, alpha=0.5, color='green')
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    ax2.set_xlabel("Valor real")
    ax2.set_ylabel("Predicción")
    ax2.set_title(f"{nombre_modelo} - Implementación sklearn")
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()