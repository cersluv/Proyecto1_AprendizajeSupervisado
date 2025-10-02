"""
Ejemplo de uso en otro archivo:

# Opción 1: Usar la función directamente
from preprocesar_datos import preparar_datos
X_train_norm, X_test_norm, X_train_pca, X_test_pca, y_train, y_test, pca, scaler = preparar_datos()

# Usar para Random Forest (mejor resultado: RF-200 con original_norm)
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train_norm, y_train)
y_pred_rf = rf_model.predict(X_test_norm)

# Usar para KNN (mejor resultado: KNN-5 con PCA)
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train_pca, y_train)
y_pred_knn = knn_model.predict(X_test_pca)

# Opción 2: Cargar datos guardados
from preprocesar_datos import cargar_datos_preprocesados
X_train_norm, X_test_norm, X_train_pca, X_test_pca, y_train, y_test, pca, scaler = cargar_datos_preprocesados()
"""