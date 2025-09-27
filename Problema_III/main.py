# main.py - Problema 3

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import time

# Importar módulos propios
from preprocessing import preparar_datos
from model import knn_sklearn, random_forest_sklearn, knn_custom, random_forest_custom, predecir

def evaluar_modelo(y_true, y_pred, nombre_modelo):
    """
    Evalúa el rendimiento del modelo con múltiples métricas
    """
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')
    
    print(f"\n{'='*50}")
    print(f"RESULTADOS - {nombre_modelo}")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Score (Macro): {f1_macro:.4f}")
    print(f"F1-Score (Weighted): {f1_weighted:.4f}")
    
    return accuracy, f1_macro, f1_weighted

def mostrar_reporte_detallado(y_true, y_pred, nombre_modelo):
    """
    Muestra un reporte detallado de clasificación
    """
    print(f"\nReporte detallado - {nombre_modelo}:")
    print(classification_report(y_true, y_pred))
    
    print(f"\nMatriz de confusión - {nombre_modelo}:")
    print(confusion_matrix(y_true, y_pred))

def main():
    """
    Función principal que ejecuta el pipeline completo
    """
    print("="*70)
    print("PROBLEMA III - CLASIFICACIÓN DE IMÁGENES")
    print("Comparación: Sklearn vs Implementaciones Propias")
    print("="*70)
    
    # --------------------------
    # 1. Preprocesamiento
    # --------------------------
    print("\n1. PREPROCESAMIENTO DE DATOS")
    print("-" * 50)
    
    start_time = time.time()
    X_train_norm, X_test_norm, X_train_pca, X_test_pca, y_train, y_test, pca, scaler = preparar_datos()
    preprocessing_time = time.time() - start_time
    
    print(f"Tiempo de preprocesamiento: {preprocessing_time:.2f} segundos")
    
    # Almacenar resultados
    resultados = {}
    
    # --------------------------
    # 2. Random Forest - Sklearn
    # --------------------------
    print("\n2. RANDOM FOREST SKLEARN (200 estimadores + datos normalizados)")
    print("-" * 50)
    
    print("Entrenando Random Forest Sklearn...")
    start_time = time.time()
    rf_sklearn_model = random_forest_sklearn(X_train_norm, y_train, n_estimators=200)
    rf_sklearn_train_time = time.time() - start_time
    
    print("Realizando predicciones...")
    start_time = time.time()
    y_pred_rf_sklearn = predecir(rf_sklearn_model, X_test_norm)
    rf_sklearn_pred_time = time.time() - start_time
    
    print(f"Tiempo de entrenamiento: {rf_sklearn_train_time:.2f} segundos")
    print(f"Tiempo de predicción: {rf_sklearn_pred_time:.2f} segundos")
    
    # Evaluar Random Forest Sklearn
    rf_sklearn_acc, rf_sklearn_f1_macro, rf_sklearn_f1_weighted = evaluar_modelo(
        y_test, y_pred_rf_sklearn, "RANDOM FOREST SKLEARN"
    )
    
    resultados['rf_sklearn'] = {
        'model': rf_sklearn_model,
        'accuracy': rf_sklearn_acc,
        'f1_macro': rf_sklearn_f1_macro,
        'train_time': rf_sklearn_train_time,
        'pred_time': rf_sklearn_pred_time,
        'predictions': y_pred_rf_sklearn
    }
    
    # --------------------------
    # 3. Random Forest - Custom
    # --------------------------
    # print("\n3. RANDOM FOREST CUSTOM (200 estimadores + datos normalizados)")
    # print("-" * 50)
    
    # print("Entrenando Random Forest Custom...")
    # start_time = time.time()
    # rf_custom_model = random_forest_custom(X_train_norm, y_train, n_estimators=200)
    # rf_custom_train_time = time.time() - start_time
    
    # print("Realizando predicciones...")
    # start_time = time.time()
    # y_pred_rf_custom = predecir(rf_custom_model, X_test_norm)
    # rf_custom_pred_time = time.time() - start_time
    
    # print(f"Tiempo de entrenamiento: {rf_custom_train_time:.2f} segundos")
    # print(f"Tiempo de predicción: {rf_custom_pred_time:.2f} segundos")
    
    # # Evaluar Random Forest Custom
    # rf_custom_acc, rf_custom_f1_macro, rf_custom_f1_weighted = evaluar_modelo(
    #     y_test, y_pred_rf_custom, "RANDOM FOREST CUSTOM"
    # )
    
    # resultados['rf_custom'] = {
    #     'model': rf_custom_model,
    #     'accuracy': rf_custom_acc,
    #     'f1_macro': rf_custom_f1_macro,
    #     'train_time': rf_custom_train_time,
    #     'pred_time': rf_custom_pred_time,
    #     'predictions': y_pred_rf_custom
    # }
    
    # --------------------------
    # 4. KNN - Sklearn
    # --------------------------
    print("\n4. KNN SKLEARN (k=5 + datos PCA)")
    print("-" * 50)
    
    print("Entrenando KNN Sklearn...")
    start_time = time.time()
    knn_sklearn_model = knn_sklearn(X_train_pca, y_train, k=5)
    knn_sklearn_train_time = time.time() - start_time
    
    print("Realizando predicciones...")
    start_time = time.time()
    y_pred_knn_sklearn = predecir(knn_sklearn_model, X_test_pca)
    knn_sklearn_pred_time = time.time() - start_time
    
    print(f"Tiempo de entrenamiento: {knn_sklearn_train_time:.2f} segundos")
    print(f"Tiempo de predicción: {knn_sklearn_pred_time:.2f} segundos")
    
    # Evaluar KNN Sklearn
    knn_sklearn_acc, knn_sklearn_f1_macro, knn_sklearn_f1_weighted = evaluar_modelo(
        y_test, y_pred_knn_sklearn, "KNN SKLEARN"
    )
    
    resultados['knn_sklearn'] = {
        'model': knn_sklearn_model,
        'accuracy': knn_sklearn_acc,
        'f1_macro': knn_sklearn_f1_macro,
        'train_time': knn_sklearn_train_time,
        'pred_time': knn_sklearn_pred_time,
        'predictions': y_pred_knn_sklearn
    }
    
    # --------------------------
    # 5. KNN - Custom
    # --------------------------
    print("\n5. KNN CUSTOM (k=5 + datos PCA)")
    print("-" * 50)
    
    print("Entrenando KNN Custom...")
    start_time = time.time()
    knn_custom_model = knn_custom(X_train_pca, y_train, k=5)
    knn_custom_train_time = time.time() - start_time
    
    print("Realizando predicciones...")
    start_time = time.time()
    y_pred_knn_custom = predecir(knn_custom_model, X_test_pca)
    knn_custom_pred_time = time.time() - start_time
    
    print(f"Tiempo de entrenamiento: {knn_custom_train_time:.2f} segundos")
    print(f"Tiempo de predicción: {knn_custom_pred_time:.2f} segundos")
    
    # Evaluar KNN Custom
    knn_custom_acc, knn_custom_f1_macro, knn_custom_f1_weighted = evaluar_modelo(
        y_test, y_pred_knn_custom, "KNN CUSTOM"
    )
    
    resultados['knn_custom'] = {
        'model': knn_custom_model,
        'accuracy': knn_custom_acc,
        'f1_macro': knn_custom_f1_macro,
        'train_time': knn_custom_train_time,
        'pred_time': knn_custom_pred_time,
        'predictions': y_pred_knn_custom
    }
    
    # --------------------------
    # 6. Comparación final
    # --------------------------
    print("\n6. COMPARACIÓN FINAL")
    print("-" * 70)
    
    print(f"{'Modelo':<18} {'Accuracy':<10} {'F1-Macro':<10} {'Entrena (s)':<12} {'Predice (s)':<12}")
    print("-" * 70)
    
    modelos_orden = ['rf_sklearn', 'rf_custom', 'knn_sklearn', 'knn_custom']
    nombres_modelos = {
        'rf_sklearn': 'RF Sklearn',
        'rf_custom': 'RF Custom', 
        'knn_sklearn': 'KNN Sklearn',
        'knn_custom': 'KNN Custom'
    }
    
    for modelo_key in modelos_orden:
        r = resultados[modelo_key]
        nombre = nombres_modelos[modelo_key]
        print(f"{nombre:<18} {r['accuracy']:<10.4f} {r['f1_macro']:<10.4f} {r['train_time']:<12.2f} {r['pred_time']:<12.2f}")
    
    # Encontrar el mejor modelo
    mejor_accuracy = max(resultados.values(), key=lambda x: x['accuracy'])['accuracy']
    mejor_modelo = [k for k, v in resultados.items() if v['accuracy'] == mejor_accuracy][0]
    
    print(f"\n🏆 MEJOR MODELO: {nombres_modelos[mejor_modelo]} (Accuracy: {mejor_accuracy:.4f})")
    
    # --------------------------
    # 7. Análisis comparativo
    # --------------------------
    print("\n7. ANÁLISIS COMPARATIVO")
    print("-" * 50)
    
    # Comparar implementaciones del mismo algoritmo
    print("Random Forest:")
    rf_diff_acc = abs(resultados['rf_sklearn']['accuracy'] - resultados['rf_custom']['accuracy'])
    rf_diff_time = resultados['rf_custom']['train_time'] - resultados['rf_sklearn']['train_time']
    print(f"  Diferencia en Accuracy: {rf_diff_acc:.4f}")
    print(f"  Diferencia en tiempo entrenamiento: {rf_diff_time:+.2f}s")
    
    print("\nKNN:")
    knn_diff_acc = abs(resultados['knn_sklearn']['accuracy'] - resultados['knn_custom']['accuracy'])
    knn_diff_time = resultados['knn_custom']['pred_time'] - resultados['knn_sklearn']['pred_time']
    print(f"  Diferencia en Accuracy: {knn_diff_acc:.4f}")
    print(f"  Diferencia en tiempo predicción: {knn_diff_time:+.2f}s")
    
    # --------------------------
    # 8. Reportes detallados (opcional)
    # --------------------------
    respuesta = input("\n¿Mostrar reportes detallados de clasificación? (s/n): ").lower().strip()
    if respuesta == 's':
        for modelo_key in modelos_orden:
            nombre = nombres_modelos[modelo_key]
            preds = resultados[modelo_key]['predictions']
            mostrar_reporte_detallado(y_test, preds, nombre)
    
    print("\n" + "="*70)
    print("COMPARACIÓN SKLEARN vs CUSTOM COMPLETADA")
    print("="*70)
    
    return resultados

if __name__ == "__main__":
    resultados = main()