import pandas as pd
import numpy as np
import cv2
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Rutas
train_path = "Problema_III/dataset/Train.csv"
test_path = "Problema_III/dataset/Test.csv"

def load_data(path):
    df = pd.read_csv(path)
    labels = df.iloc[:, 0].values.astype(np.int32)
    images = df.iloc[:, 1:].values.astype(np.uint8)
    return images, labels

def binarize_image(img, threshold=0):
    return np.where(img > threshold, 255, 0).astype(np.uint8)

def apply_clahe(img, clip_limit=2.0, tile_grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(img)

def apply_gaussian_blur(img, kernel_size=3):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def apply_median_filter(img, kernel_size=3):
    return cv2.medianBlur(img, kernel_size)

def apply_edge_detection(img, method='sobel'):
    if method == 'sobel':
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.sqrt(grad_x**2 + grad_y**2).astype(np.uint8)
    elif method == 'canny':
        return cv2.Canny(img, 50, 150)

def apply_morphological_ops(img, operation='closing', kernel_size=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if operation == 'opening':
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif operation == 'closing':
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    elif operation == 'gradient':
        return cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

def standardize_data(X_train, X_test):
    """Estandarización usando estadísticas del conjunto de entrenamiento"""
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    return X_train_std, X_test_std, scaler

def apply_pca(X_train, X_test, n_components=200):
    """Aplicar PCA para reducción de dimensionalidad"""
    pca = PCA(n_components=n_components)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    print(f"Varianza explicada con {n_components} componentes: {pca.explained_variance_ratio_.sum():.4f}")
    return X_train_pca, X_test_pca, pca

def select_best_features(X_train, X_test, y_train, k=400):
    """Selección de las mejores k características"""
    selector = SelectKBest(score_func=chi2, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected, selector

def create_feature_combinations(X_original, X_processed_list):
    """Combinar diferentes tipos de características procesadas"""
    combined = X_original
    for X_proc in X_processed_list:
        combined = np.hstack([combined, X_proc])
    return combined

def preprocess_and_save():
    # Cargar datos
    print("Cargando datos...")
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)
    
    print(f"Datos cargados: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} de prueba")
    
    # Normalización básica primero
    X_train_norm = X_train / 255.0
    X_test_norm = X_test / 255.0
    
    # Procesamiento de imágenes
    print("Aplicando transformaciones de imagen...")
    
    # 1. Binarización
    X_train_bin = np.array([binarize_image(img.reshape(28,28)).flatten() for img in X_train])
    X_test_bin = np.array([binarize_image(img.reshape(28,28)).flatten() for img in X_test])
    
    # 2. CLAHE
    X_train_clahe = np.array([apply_clahe(img.reshape(28,28)).flatten() for img in X_train])
    X_test_clahe = np.array([apply_clahe(img.reshape(28,28)).flatten() for img in X_test])
    
    # 3. Filtro Gaussiano
    X_train_gauss = np.array([apply_gaussian_blur(img.reshape(28,28)).flatten() for img in X_train])
    X_test_gauss = np.array([apply_gaussian_blur(img.reshape(28,28)).flatten() for img in X_test])
    
    # 4. Detección de bordes Sobel
    X_train_edges = np.array([apply_edge_detection(img.reshape(28,28), 'sobel').flatten() for img in X_train])
    X_test_edges = np.array([apply_edge_detection(img.reshape(28,28), 'sobel').flatten() for img in X_test])
    
    # 5. Operaciones morfológicas
    X_train_morph = np.array([apply_morphological_ops(img.reshape(28,28), 'closing').flatten() for img in X_train])
    X_test_morph = np.array([apply_morphological_ops(img.reshape(28,28), 'closing').flatten() for img in X_test])
    
    # Normalizar todas las versiones procesadas
    datasets_to_normalize = [
        ("bin", X_train_bin, X_test_bin),
        ("clahe", X_train_clahe, X_test_clahe), 
        ("gauss", X_train_gauss, X_test_gauss),
        ("edges", X_train_edges, X_test_edges),
        ("morph", X_train_morph, X_test_morph)
    ]
    
    processed_datasets = {}
    
    for name, train_data, test_data in datasets_to_normalize:
        # Normalizar a [0,1]
        train_norm = train_data / 255.0
        test_norm = test_data / 255.0
        processed_datasets[name] = (train_norm, test_norm)
    
    # Estandarización
    print("Aplicando estandarización...")
    X_train_std, X_test_std, scaler = standardize_data(X_train_norm, X_test_norm)
    
    # Combinación de características
    print("Creando combinaciones de características...")
    
    # Combinar original + CLAHE + bordes
    X_train_combined = create_feature_combinations(
        X_train_norm, 
        [processed_datasets["clahe"][0], processed_datasets["edges"][0]]
    )
    X_test_combined = create_feature_combinations(
        X_test_norm, 
        [processed_datasets["clahe"][1], processed_datasets["edges"][1]]
    )
    
    # Reducción de dimensionalidad
    print("Aplicando PCA...")
    X_train_pca, X_test_pca, pca = apply_pca(X_train_std, X_test_std, n_components=200)
    
    # Selección de características con datos binarizados (mejor para chi2)
    print("Seleccionando mejores características...")
    X_train_selected, X_test_selected, selector = select_best_features(
        processed_datasets["bin"][0], processed_datasets["bin"][1], y_train, k=400
    )
    
    # Guardar todos los conjuntos procesados
    datasets_to_save = {
        "original_norm": (X_train_norm, X_test_norm),
        "standardized": (X_train_std, X_test_std),
        "binary": processed_datasets["bin"],
        "clahe": processed_datasets["clahe"],
        "gaussian": processed_datasets["gauss"],
        "edges": processed_datasets["edges"],
        "morphological": processed_datasets["morph"],
        "combined": (X_train_combined, X_test_combined),
        "pca": (X_train_pca, X_test_pca),
        "selected": (X_train_selected, X_test_selected)
    }
    
    print("Guardando datos procesados...")
    for name, (train_data, test_data) in datasets_to_save.items():
        np.save(f"train_{name}.npy", train_data)
        np.save(f"test_{name}.npy", test_data)
    
    # Guardar etiquetas y objetos de transformación
    np.save("train_labels.npy", y_train)
    np.save("test_labels.npy", y_test)
    
    # Guardar transformadores para uso posterior
    import joblib
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(pca, "pca.pkl")
    joblib.dump(selector, "feature_selector.pkl")
    
    print("Preprocesamiento completado!")
    print(f"Conjuntos de datos creados: {list(datasets_to_save.keys())}")
    print(f"Formas de los datos:")
    for name, (train_data, _) in datasets_to_save.items():
        print(f"  {name}: {train_data.shape}")

def visualize_preprocessing_examples(num_samples=5):
    """Visualizar ejemplos del preprocesamiento"""
    X_train, y_train = load_data(train_path)
    
    # Tomar algunas muestras aleatorias
    indices = np.random.choice(X_train.shape[0], num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 6, figsize=(15, 3*num_samples))
    
    processing_functions = [
        ("Original", lambda x: x),
        ("Binarizada", lambda x: binarize_image(x)),
        ("CLAHE", lambda x: apply_clahe(x)),
        ("Gaussiano", lambda x: apply_gaussian_blur(x)),
        ("Bordes", lambda x: apply_edge_detection(x, 'sobel')),
        ("Morfológica", lambda x: apply_morphological_ops(x, 'closing'))
    ]
    
    for i, idx in enumerate(indices):
        img = X_train[idx].reshape(28, 28)
        
        for j, (title, func) in enumerate(processing_functions):
            processed_img = func(img.copy())
            axes[i, j].imshow(processed_img, cmap='gray')
            axes[i, j].set_title(f"{title}\nClase: {y_train[idx]}")
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig('preprocessing_examples.png', dpi=150, bbox_inches='tight')
    plt.show()

def main_evaluation():
    """Función principal mejorada para evaluar todos los conjuntos de datos"""
    import numpy as np
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    import time
    
    def evaluate_model(model, X_train, y_train, X_test, y_test, model_name, data_name):
        print(f"\nEntrenando {model_name} con {data_name}...")
        start_time = time.time()
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        train_time = time.time() - start_time
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
        
        return acc, f1, train_time
    
    # Cargar etiquetas
    y_train = np.load("train_labels.npy")
    y_test = np.load("test_labels.npy")
    
    # Definir conjuntos de datos a evaluar
    datasets = {
        "original_norm": "Datos originales normalizados",
        "standardized": "Datos estandarizados", 
        "binary": "Datos binarizados",
        "clahe": "CLAHE aplicado",
        "gaussian": "Filtro Gaussiano",
        "edges": "Detección de bordes",
        "combined": "Características combinadas",
        "pca": "PCA (200 componentes)",
        "selected": "Características seleccionadas"
    }
    
    # Configurar modelos con diferentes hiperparámetros
    models = {
        "KNN-5": KNeighborsClassifier(n_neighbors=5),
        "KNN-7": KNeighborsClassifier(n_neighbors=7),
        "KNN-9": KNeighborsClassifier(n_neighbors=9),
        "RF-100": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "RF-200": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    }
    
    results = []
    
    print("="*60)
    print("EVALUACIÓN COMPLETA DE MODELOS")
    print("="*60)
    
    for data_name, data_desc in datasets.items():
        try:
            # Cargar datos
            X_train = np.load(f"train_{data_name}.npy")
            X_test = np.load(f"test_{data_name}.npy")
            
            print(f"\n{'='*20} {data_desc.upper()} {'='*20}")
            print(f"Shape: {X_train.shape}")
            
            for model_name, model in models.items():
                try:
                    acc, f1, train_time = evaluate_model(
                        model, X_train, y_train, X_test, y_test, 
                        model_name, data_name
                    )
                    
                    results.append({
                        'dataset': data_name,
                        'model': model_name,
                        'accuracy': acc,
                        'f1_macro': f1,
                        'train_time': train_time
                    })
                    
                    print(f"{model_name:8} -> Acc: {acc:.4f}, F1: {f1:.4f}, Tiempo: {train_time:.2f}s")
                    
                except Exception as e:
                    print(f"Error con {model_name}: {e}")
                    
        except FileNotFoundError:
            print(f"Archivo no encontrado para {data_name}")
    
    # Mostrar mejores resultados
    print("\n" + "="*60)
    print("MEJORES RESULTADOS")
    print("="*60)
    
    if results:
        # Convertir a DataFrame para análisis más fácil
        import pandas as pd
        results_df = pd.DataFrame(results)
        
        # Mejores por accuracy
        print("\nTOP 5 - ACCURACY:")
        top_acc = results_df.nlargest(5, 'accuracy')
        for _, row in top_acc.iterrows():
            print(f"{row['model']:8} + {row['dataset']:15} -> Acc: {row['accuracy']:.4f}")
        
        # Mejores por F1
        print("\nTOP 5 - F1 MACRO:")
        top_f1 = results_df.nlargest(5, 'f1_macro')
        for _, row in top_f1.iterrows():
            print(f"{row['model']:8} + {row['dataset']:15} -> F1: {row['f1_macro']:.4f}")
        
        # Análisis por modelo
        print("\nRESUMEN POR MODELO:")
        model_summary = results_df.groupby('model').agg({
            'accuracy': ['mean', 'max'],
            'f1_macro': ['mean', 'max'],
            'train_time': 'mean'
        }).round(4)
        print(model_summary)
        
        # Análisis por dataset
        print("\nRESUMEN POR DATASET:")
        dataset_summary = results_df.groupby('dataset').agg({
            'accuracy': ['mean', 'max'],
            'f1_macro': ['mean', 'max']
        }).round(4)
        print(dataset_summary)
        
        # Guardar resultados completos
        results_df.to_csv("model_evaluation_results.csv", index=False)
        print("\nResultados guardados en 'model_evaluation_results.csv'")

if __name__ == "__main__":
    # Primero ejecutar el preprocesamiento
    preprocess_and_save()
    
    # Luego la evaluación completa
    main_evaluation()