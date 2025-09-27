import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def preparar_datos(train_path="Problema_III/dataset/Train.csv", 
                   test_path="Problema_III/dataset/Test.csv"):
    """
    Preprocesa los datos para los dos mejores enfoques identificados:
    1. PCA (200 componentes) + KNN-5
    2. Original normalizado + RF-200
    
    Returns:
        tuple: (X_train_norm, X_test_norm, X_train_pca, X_test_pca, y_train, y_test, pca_transformer, scaler)
    """
    
    print("Cargando datos...")
    
    # -------------------------
    # Cargar datos
    # -------------------------
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    
    # Separar etiquetas y características
    y_train = train_df.iloc[:, 0].values.astype(np.int32)
    y_test = test_df.iloc[:, 0].values.astype(np.int32)
    
    X_train = train_df.iloc[:, 1:].values.astype(np.uint8)
    X_test = test_df.iloc[:, 1:].values.astype(np.uint8)
    
    print(f"Datos cargados: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} de prueba")
    print(f"Dimensiones: {X_train.shape[1]} características por imagen")
    
    # -------------------------
    # Preprocesamiento 1: Normalización básica (para RF-200)
    # -------------------------
    print("Aplicando normalización...")
    X_train_norm = X_train / 255.0
    X_test_norm = X_test / 255.0
    


    # -------------------------
    # Preprocesamiento 2: Estandarización + PCA (para KNN-5)
    # -------------------------
    
    print("Aplicando estandarización...")
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train_norm)
    X_test_std = scaler.transform(X_test_norm)
    
    print("Aplicando PCA...")
    pca = PCA(n_components=200)
    X_train_pca = pca.fit_transform(X_train_std)
    X_test_pca = pca.transform(X_test_std)
    
    print(f"Varianza explicada con 200 componentes PCA: {pca.explained_variance_ratio_.sum():.4f}")
    
    # -------------------------
    # Información de salida
    # -------------------------
    print("\nDatos preprocesados exitosamente!")
    print(f"Forma datos normalizados: {X_train_norm.shape}")
    print(f"Forma datos PCA: {X_train_pca.shape}")
    print(f"Clases únicas: {np.unique(y_train)}")
    print(f"Distribución de clases entrenamiento: {np.bincount(y_train)}")
    
    return X_train_norm, X_test_norm, X_train_pca, X_test_pca, y_train, y_test, pca, scaler

def guardar_datos_preprocesados(X_train_norm, X_test_norm, X_train_pca, X_test_pca, 
                               y_train, y_test, pca, scaler, carpeta_salida="datos_preprocesados"):
    """
    Guarda los datos preprocesados en archivos .npy para uso posterior
    """
    import os
    
    # Crear carpeta si no existe
    os.makedirs(carpeta_salida, exist_ok=True)
    
    # Guardar datos
    np.save(os.path.join(carpeta_salida, "X_train_norm.npy"), X_train_norm)
    np.save(os.path.join(carpeta_salida, "X_test_norm.npy"), X_test_norm)
    np.save(os.path.join(carpeta_salida, "X_train_pca.npy"), X_train_pca)
    np.save(os.path.join(carpeta_salida, "X_test_pca.npy"), X_test_pca)
    np.save(os.path.join(carpeta_salida, "y_train.npy"), y_train)
    np.save(os.path.join(carpeta_salida, "y_test.npy"), y_test)
    
    # Guardar transformadores
    import joblib
    joblib.dump(pca, os.path.join(carpeta_salida, "pca_transformer.pkl"))
    joblib.dump(scaler, os.path.join(carpeta_salida, "scaler.pkl"))
    
    print(f"\nDatos guardados en la carpeta: {carpeta_salida}")
    print("Archivos creados:")
    print("- X_train_norm.npy, X_test_norm.npy (para Random Forest)")
    print("- X_train_pca.npy, X_test_pca.npy (para KNN)")
    print("- y_train.npy, y_test.npy (etiquetas)")
    print("- pca_transformer.pkl, scaler.pkl (transformadores)")

def cargar_datos_preprocesados(carpeta="datos_preprocesados"):
    """
    Carga los datos preprocesados desde archivos guardados
    """
    import os
    import joblib
    
    X_train_norm = np.load(os.path.join(carpeta, "X_train_norm.npy"))
    X_test_norm = np.load(os.path.join(carpeta, "X_test_norm.npy"))
    X_train_pca = np.load(os.path.join(carpeta, "X_train_pca.npy"))
    X_test_pca = np.load(os.path.join(carpeta, "X_test_pca.npy"))
    y_train = np.load(os.path.join(carpeta, "y_train.npy"))
    y_test = np.load(os.path.join(carpeta, "y_test.npy"))
    
    pca = joblib.load(os.path.join(carpeta, "pca_transformer.pkl"))
    scaler = joblib.load(os.path.join(carpeta, "scaler.pkl"))
    
    print(f"Datos cargados desde: {carpeta}")
    print(f"Forma datos normalizados: {X_train_norm.shape}")
    print(f"Forma datos PCA: {X_train_pca.shape}")
    
    return X_train_norm, X_test_norm, X_train_pca, X_test_pca, y_train, y_test, pca, scaler

# Función principal
if __name__ == "__main__":
    # Preparar datos
    X_train_norm, X_test_norm, X_train_pca, X_test_pca, y_train, y_test, pca, scaler = preparar_datos()
    
    # Opcional: guardar datos para uso posterior
    #guardar_datos_preprocesados(X_train_norm, X_test_norm, X_train_pca, X_test_pca, 
    #                           y_train, y_test, pca, scaler)


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