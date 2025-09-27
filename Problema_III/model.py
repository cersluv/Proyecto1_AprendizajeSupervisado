# model.py - Problema 3

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# Implementación propia
# --------------------------

import numpy as np

def knn_custom(X_train, y_train, k=5):
    """
    Implementación propia optimizada de K-Nearest Neighbors con procesamiento por lotes
    """
    class KNNCustom:
        def __init__(self, k=5):
            self.k = k
            self.X_train = None
            self.y_train = None
        
        def fit(self, X_train, y_train):
            # KNN es un algoritmo lazy - solo guarda los datos
            self.X_train = X_train.copy()
            self.y_train = y_train.copy()
            return self
        
        def _euclidean_distance_batch(self, X_test_batch):
            """
            Calcula distancias euclidanas para un lote pequeño de muestras
            """
            # Shape: (batch_size, n_train)
            distances = np.sqrt(np.sum((X_test_batch[:, np.newaxis] - self.X_train[np.newaxis, :]) ** 2, axis=2))
            return distances
        
        def predict(self, X_test, batch_size=50):
            """
            Predice las clases procesando por lotes para evitar problemas de memoria
            """
            n_test = X_test.shape[0]
            predictions = []
            
            print(f"Procesando {n_test} muestras en lotes de {batch_size}...")
            
            # Procesar en lotes
            for start_idx in range(0, n_test, batch_size):
                end_idx = min(start_idx + batch_size, n_test)
                X_batch = X_test[start_idx:end_idx]
                
                # Calcular distancias para este lote
                distances_batch = self._euclidean_distance_batch(X_batch)
                
                # Para cada muestra en el lote
                for i, distances in enumerate(distances_batch):
                    # Encontrar los k vecinos más cercanos
                    nearest_indices = np.argpartition(distances, self.k)[:self.k]
                    
                    # Obtener las etiquetas de los vecinos más cercanos
                    nearest_labels = self.y_train[nearest_indices]
                    
                    # Votación por mayoría
                    from collections import Counter
                    prediction = Counter(nearest_labels).most_common(1)[0][0]
                    predictions.append(prediction)
                
                # Mostrar progreso
                processed = min(end_idx, n_test)
                if processed % (batch_size * 10) == 0 or processed == n_test:
                    print(f"  Procesadas {processed}/{n_test} muestras ({processed/n_test*100:.1f}%)")
            
            return np.array(predictions)
    
    # Crear y entrenar el modelo
    model = KNNCustom(k=k)
    model.fit(X_train, y_train)
    return model

def random_forest_custom(X_train, y_train, n_estimators=200, max_depth=10, min_samples_split=5):
    """
    Implementación propia simplificada de Random Forest
    """
    class DecisionTreeCustom:
        def __init__(self, max_depth=10, min_samples_split=5):
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.tree = None
        
        def _gini_impurity(self, y):
            """Calcula la impureza de Gini"""
            if len(y) == 0:
                return 0
            
            classes, counts = np.unique(y, return_counts=True)
            probabilities = counts / len(y)
            gini = 1 - np.sum(probabilities ** 2)
            return gini
        
        def _information_gain(self, y, left_mask):
            """Calcula la ganancia de información"""
            if len(y) == 0:
                return 0
                
            # Impureza padre
            parent_gini = self._gini_impurity(y)
            
            # Dividir datos
            left_y = y[left_mask]
            right_y = y[~left_mask]
            
            if len(left_y) == 0 or len(right_y) == 0:
                return 0
            
            # Impureza ponderada de los hijos
            n = len(y)
            left_weight = len(left_y) / n
            right_weight = len(right_y) / n
            
            weighted_gini = (left_weight * self._gini_impurity(left_y) + 
                           right_weight * self._gini_impurity(right_y))
            
            return parent_gini - weighted_gini
        
        def _best_split(self, X, y):
            """Encuentra la mejor división"""
            best_gain = 0
            best_feature = None
            best_threshold = None
            
            n_features = X.shape[1]
            
            # Probar un subconjunto aleatorio de características (feature bagging)
            n_features_to_try = int(np.sqrt(n_features))
            features_to_try = np.random.choice(n_features, n_features_to_try, replace=False)
            
            for feature in features_to_try:
                # Probar algunos valores de threshold
                feature_values = X[:, feature]
                thresholds = np.percentile(feature_values, [25, 50, 75])
                
                for threshold in thresholds:
                    left_mask = feature_values <= threshold
                    
                    if np.sum(left_mask) == 0 or np.sum(left_mask) == len(y):
                        continue
                    
                    gain = self._information_gain(y, left_mask)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_feature = feature
                        best_threshold = threshold
            
            return best_feature, best_threshold, best_gain
        
        def _build_tree(self, X, y, depth=0):
            """Construye el árbol recursivamente"""
            # Casos base
            if (depth >= self.max_depth or 
                len(y) < self.min_samples_split or 
                len(np.unique(y)) == 1):
                
                # Retornar hoja con la clase más común
                from collections import Counter
                return Counter(y).most_common(1)[0][0]
            
            # Encontrar la mejor división
            feature, threshold, gain = self._best_split(X, y)
            
            if feature is None or gain == 0:
                from collections import Counter
                return Counter(y).most_common(1)[0][0]
            
            # Dividir datos
            left_mask = X[:, feature] <= threshold
            
            # Crear nodo
            node = {
                'feature': feature,
                'threshold': threshold,
                'left': self._build_tree(X[left_mask], y[left_mask], depth + 1),
                'right': self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
            }
            
            return node
        
        def fit(self, X, y):
            """Entrena el árbol"""
            self.tree = self._build_tree(X, y)
            return self
        
        def _predict_single(self, x):
            """Predice para una muestra"""
            node = self.tree
            
            while isinstance(node, dict):
                if x[node['feature']] <= node['threshold']:
                    node = node['left']
                else:
                    node = node['right']
            
            return node
        
        def predict(self, X):
            """Predice para múltiples muestras"""
            return np.array([self._predict_single(x) for x in X])
    
    class RandomForestCustom:
        def __init__(self, n_estimators=200, max_depth=10, min_samples_split=5):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.min_samples_split = min_samples_split
            self.trees = []
        
        def fit(self, X, y):
            """Entrena el bosque"""
            self.trees = []
            n_samples = X.shape[0]
            
            for i in range(self.n_estimators):
                # Bootstrap sampling
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_bootstrap = X[indices]
                y_bootstrap = y[indices]
                
                # Entrenar árbol
                tree = DecisionTreeCustom(
                    max_depth=self.max_depth,
                    min_samples_split=self.min_samples_split
                )
                tree.fit(X_bootstrap, y_bootstrap)
                self.trees.append(tree)
                
                # Mostrar progreso
                if (i + 1) % 50 == 0:
                    print(f"Entrenado {i + 1}/{self.n_estimators} árboles...")
            
            return self
        
        def predict(self, X):
            """Predice usando votación por mayoría"""
            # Obtener predicciones de todos los árboles
            tree_predictions = np.array([tree.predict(X) for tree in self.trees])
            
            # Votación por mayoría
            predictions = []
            for i in range(X.shape[0]):
                votes = tree_predictions[:, i]
                from collections import Counter
                prediction = Counter(votes).most_common(1)[0][0]
                predictions.append(prediction)
            
            return np.array(predictions)
    
    # Crear y entrenar el modelo
    print(f"Creando Random Forest con {n_estimators} árboles...")
    model = RandomForestCustom(n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split)
    model.fit(X_train, y_train)
    return model

# --------------------------
# Implementación con sklearn
# --------------------------

def knn_sklearn(X_train, y_train, k=5):
    """
    KNN con sklearn usando los hiperparámetros óptimos encontrados
    Mejor resultado: k=5 con datos PCA (Accuracy: 0.8699)
    """
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def random_forest_sklearn(X_train, y_train, n_estimators=200, random_state=42):
    """
    Random Forest con sklearn usando los hiperparámetros óptimos encontrados
    Mejor resultado: 200 estimadores con datos normalizados (Accuracy: 0.8875)
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1  # Usar todos los cores disponibles
    )
    model.fit(X_train, y_train)
    return model

# --------------------------
# Predicción
# --------------------------

def predecir(model, X_test):
    """
    Realizar predicciones usando el modelo entrenado
    """
    return model.predict(X_test)

def predecir_probabilidades(model, X_test):
    """
    Obtener probabilidades de predicción (útil para análisis más detallado)
    """
    try:
        return model.predict_proba(X_test)
    except AttributeError:
        print("El modelo no soporta predict_proba")
        return None