import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from collections import Counter

# --------------------------
# Implementación propia
# --------------------------

class LogisticRegressionCustom:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def _sigmoid(self, z):
        """Función sigmoide con protección contra overflow"""
        # Clipear z para evitar overflow en exp
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def _compute_cost(self, y_true, y_pred):
        """Calcula la función de costo (log-loss)"""
        # Evitar log(0) añadiendo epsilon
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def fit(self, X, y):
        """Entrena el modelo usando gradiente descendente"""
        n_samples, n_features = X.shape
        
        # Inicializar pesos y bias
        self.weights = np.random.normal(0, 0.01, n_features)
        self.bias = 0
        
        # Gradiente descendente
        for i in range(self.max_iterations):
            # Forward pass
            linear_pred = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_pred)
            
            # Calcular costo
            cost = self._compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Calcular gradientes
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.mean(y_pred - y)
            
            # Actualizar parámetros
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Criterio de convergencia
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tolerance:
                break
                
        return self
    
    def predict_proba(self, X):
        """Predice probabilidades"""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_pred)
    
    def predict(self, X):
        """Predice clases (0 o 1)"""
        probabilities = self.predict_proba(X)
        return (probabilities >= 0.5).astype(int)


class DecisionTreeCustom:
    def __init__(self, max_depth=None, min_samples_split=2, min_samples_leaf=1):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree = None
        
    def _gini_impurity(self, y):
        """Calcula la impureza de Gini"""
        if len(y) == 0:
            return 0
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _information_gain(self, parent, left_child, right_child):
        """Calcula la ganancia de información"""
        weight_left = len(left_child) / len(parent)
        weight_right = len(right_child) / len(parent)
        
        gain = self._gini_impurity(parent) - (
            weight_left * self._gini_impurity(left_child) +
            weight_right * self._gini_impurity(right_child)
        )
        return gain
    
    def _best_split(self, X, y):
        """Encuentra la mejor división"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Probar cada valor único como threshold
            for threshold in unique_values:
                # Dividir datos
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) < self.min_samples_leaf or np.sum(right_mask) < self.min_samples_leaf:
                    continue
                
                # Calcular ganancia de información
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _build_tree(self, X, y, depth=0):
        """Construye el árbol recursivamente"""
        n_samples = len(y)
        
        # Condiciones de parada
        if (self.max_depth is not None and depth >= self.max_depth or
            n_samples < self.min_samples_split or
            len(np.unique(y)) == 1):
            
            # Nodo hoja: devolver la clase mayoritaria
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # Encontrar la mejor división
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        if best_feature is None or best_gain <= 0:
            # No se pudo encontrar una buena división
            leaf_value = Counter(y).most_common(1)[0][0]
            return {'leaf': True, 'value': leaf_value}
        
        # Dividir datos
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # Crear nodos hijos recursivamente
        left_subtree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_subtree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_subtree,
            'right': right_subtree
        }
    
    def fit(self, X, y):
        """Entrena el árbol de decisión"""
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_single(self, x, tree):
        """Predice una sola muestra"""
        if tree['leaf']:
            return tree['value']
        
        if x[tree['feature']] <= tree['threshold']:
            return self._predict_single(x, tree['left'])
        else:
            return self._predict_single(x, tree['right'])
    
    def predict(self, X):
        """Predice múltiples muestras"""
        return np.array([self._predict_single(x, self.tree) for x in X])


def logistic_regression_custom(X_train, y_train):
    """Implementación propia de regresión logística"""
    model = LogisticRegressionCustom(learning_rate=0.01, max_iterations=1000)
    model.fit(X_train, y_train)
    return model

def decision_tree_custom(X_train, y_train):
    """Implementación propia de árbol de decisión"""
    model = DecisionTreeCustom(max_depth=10, min_samples_split=20, min_samples_leaf=5)
    model.fit(X_train, y_train)
    return model

# --------------------------
# Implementación con sklearn
# --------------------------

def logistic_regression_sklearn(X_train, y_train):
    """Regresión logística usando sklearn"""
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model

def decision_tree_sklearn(X_train, y_train):
    """Árbol de decisión usando sklearn"""
    model = DecisionTreeClassifier(random_state=42, max_depth=10, 
                                 min_samples_split=20, min_samples_leaf=5)
    model.fit(X_train, y_train)
    return model

# --------------------------
# Predicción
# --------------------------

def predecir(model, X_test):
    """Función universal de predicción"""
    return model.predict(X_test)

# --------------------------
# GridSearch para optimizar hiperparámetros
# --------------------------

def optimize_logistic_regression_sklearn(X_train, y_train, cv=5):
    """Optimiza hiperparámetros de regresión logística con GridSearch"""
    from sklearn.model_selection import GridSearchCV
    
    # Definir grid de parámetros a probar
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularización
        'penalty': ['l1', 'l2', 'elasticnet', None],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    }
    
    # Crear modelo base
    model = LogisticRegression(random_state=42)
    
    # GridSearch con validación cruzada
    grid_search = GridSearchCV(
        model, param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=-1,  # Usar todos los cores disponibles
        verbose=1
    )
    
    # Ajustar
    grid_search.fit(X_train, y_train)
    
    print(f"Mejores parámetros Logistic Regression: {grid_search.best_params_}")
    print(f"Mejor score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def optimize_decision_tree_sklearn(X_train, y_train, cv=5):
    """Optimiza hiperparámetros de árbol de decisión con GridSearch"""
    from sklearn.model_selection import GridSearchCV
    
    # Definir grid de parámetros a probar
    param_grid = {
        'max_depth': [5, 10, 15, 20, 25, None],
        'min_samples_split': [2, 10, 20, 50],
        'min_samples_leaf': [1, 5, 10, 20],
        'criterion': ['gini', 'entropy'],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Crear modelo base
    model = DecisionTreeClassifier(random_state=42)
    
    # GridSearch con validación cruzada
    grid_search = GridSearchCV(
        model, param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    # Ajustar
    grid_search.fit(X_train, y_train)
    
    print(f"Mejores parámetros Decision Tree: {grid_search.best_params_}")
    print(f"Mejor score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def optimize_custom_models_simple(X_train, y_train, optimized_sklearn_models):
    """Optimiza modelos custom usando los parámetros encontrados por sklearn (método simple)"""
    
    log_sklearn_opt, tree_sklearn_opt = optimized_sklearn_models
    
    # ============================================
    # Optimizar Regresión Logística Custom
    # ============================================
    print("🔧 Optimizando Regresión Logística Custom...")
    
    # Grid pequeño para logistic regression custom
    learning_rates = [0.001, 0.01, 0.1]
    max_iterations = [1000, 2000]
    
    best_log_score = 0
    best_log_params = {}
    best_log_model = None
    
    print(f"🔍 Probando {len(learning_rates)} × {len(max_iterations)} = {len(learning_rates) * len(max_iterations)} combinaciones...")
    
    for lr in learning_rates:
        for max_iter in max_iterations:
            model = LogisticRegressionCustom(learning_rate=lr, max_iterations=max_iter)
            model.fit(X_train, y_train)
            
            # Evaluar en el mismo conjunto de entrenamiento (simple)
            y_pred = model.predict(X_train)
            score = np.mean(y_pred == y_train)
            
            if score > best_log_score:
                best_log_score = score
                best_log_params = {'learning_rate': lr, 'max_iterations': max_iter}
                best_log_model = model
    
    print(f"✅ Mejores parámetros Logistic Custom: {best_log_params}")
    print(f"   Score: {best_log_score:.4f}")
    
    # ============================================
    # Optimizar Árbol Custom usando parámetros de sklearn
    # ============================================
    print("\n🔧 Optimizando Árbol Custom usando parámetros de sklearn...")
    
    # Obtener los mejores parámetros del modelo sklearn optimizado
    sklearn_params = tree_sklearn_opt.get_params()
    
    print(f"📋 Parámetros sklearn encontrados:")
    print(f"   • max_depth: {sklearn_params.get('max_depth')}")
    print(f"   • min_samples_split: {sklearn_params.get('min_samples_split')}")
    print(f"   • min_samples_leaf: {sklearn_params.get('min_samples_leaf')}")
    
    # Usar exactamente los mismos parámetros de sklearn
    best_max_depth = sklearn_params.get('max_depth')
    best_min_split = sklearn_params.get('min_samples_split') 
    best_min_leaf = sklearn_params.get('min_samples_leaf')
    
    # Crear modelo custom con los parámetros de sklearn
    best_tree_model = DecisionTreeCustom(
        max_depth=best_max_depth,
        min_samples_split=best_min_split,
        min_samples_leaf=best_min_leaf
    )
    best_tree_model.fit(X_train, y_train)
    
    # Evaluar
    y_pred = best_tree_model.predict(X_train)
    tree_score = np.mean(y_pred == y_train)
    
    best_tree_params = {
        'max_depth': best_max_depth,
        'min_samples_split': best_min_split,
        'min_samples_leaf': best_min_leaf
    }
    
    print(f"✅ Parámetros Tree Custom (copiados de sklearn): {best_tree_params}")
    print(f"   Score: {tree_score:.4f}")
    print(f"   📝 Nota: Usa exactamente los mismos parámetros que sklearn encontró")
    
    return best_log_model, best_tree_model

def get_optimized_custom_models(X_train, y_train):
    """Función principal para obtener modelos custom optimizados de forma eficiente"""
    
    print("🚀 Iniciando optimización: sklearn primero, custom después...")
    
    # Primero optimizar sklearn
    print("\n1️⃣ Optimizando modelos sklearn con GridSearch...")
    log_sklearn_opt = optimize_logistic_regression_sklearn(X_train, y_train)
    tree_sklearn_opt = optimize_decision_tree_sklearn(X_train, y_train)
    
    # Luego optimizar custom usando esos parámetros como referencia
    print("\n2️⃣ Optimizando modelos custom usando parámetros de sklearn como guía...")
    log_custom_opt, tree_custom_opt = optimize_custom_models_simple(
        X_train, y_train, (log_sklearn_opt, tree_sklearn_opt)
    )
    
    print("\n✅ Optimización completa!")
    
    return log_custom_opt, tree_custom_opt, log_sklearn_opt, tree_sklearn_opt

# --------------------------
# Funciones actualizadas con optimización
# --------------------------

def logistic_regression_sklearn_optimized(X_train, y_train):
    """Regresión logística optimizada con GridSearch"""
    return optimize_logistic_regression_sklearn(X_train, y_train)

def decision_tree_sklearn_optimized(X_train, y_train):
    """Árbol de decisión optimizado con GridSearch"""
    return optimize_decision_tree_sklearn(X_train, y_train)

# --------------------------
# Función para evaluar modelos
# --------------------------

def evaluar_modelo(y_true, y_pred):
    """Calcula métricas de evaluación"""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }