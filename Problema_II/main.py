from model import (
    # Implementaciones sklearn
    linear_regression_sklearn,
    random_forest_sklearn,
    optimize_random_forest_sklearn,
    # Implementaciones manuales
    linear_regression_custom,
    random_forest_custom,
    # Funciones comunes
    predecir,
    evaluar_modelo,
    comparar_modelos
)
from preprocessing import preparar_datos  
import numpy as np 

def main():
    print("=== PROYECTO DE RECONOCIMIENTO DE PATRONES ===")
    print("=== OPTIMIZACIÓN CON GRIDSEARCH Y COMPARACIÓN EQUITATIVA ===")
    
    # --------------------------
    # Preprocesamiento
    # --------------------------
    print("\n1. PREPARANDO DATOS...")
    X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preparar_datos()
    print(f"Datos preparados: {X_train.shape[0]} muestras de entrenamiento, {X_test.shape[0]} de prueba")

    # --------------------------
    # Regresión Lineal (sin cambios, siempre es la misma)
    # --------------------------
    print("\n" + "="*60)
    print("2. REGRESIÓN LINEAL (baseline)")
    print("="*60)
    
    # Manual
    lr_custom = linear_regression_custom(X_train_scaled, y_train)
    y_pred_lr_custom = predecir(lr_custom, X_test_scaled)
    
    # sklearn
    lr_sklearn = linear_regression_sklearn(X_train_scaled, y_train)
    y_pred_lr_sklearn = predecir(lr_sklearn, X_test_scaled)
    
    # Comparar
    comparar_modelos(y_test, y_pred_lr_custom, y_pred_lr_sklearn, "Regresión Lineal")

    # --------------------------
    # Random Forest CON OPTIMIZACIÓN
    # --------------------------
    print("\n" + "="*60)
    print("3. RANDOM FOREST CON GRIDSEARCH")
    print("="*60)
    
    # PASO 1: Optimizar con sklearn y GridSearch
    print("\n3.1 Optimizando hiperparámetros con sklearn...")
    rf_optimized, best_params = optimize_random_forest_sklearn(X_train, y_train, random_state=42)
    
    # PASO 2: Usar los MISMOS parámetros en la implementación manual
    print(f"\n3.2 Aplicando los mejores parámetros a la implementación manual...")
    print(f"Parámetros optimizados: {best_params}")
    
    rf_custom_optimized = random_forest_custom(
        X_train, y_train, 
        n_estimators=best_params['n_estimators'],
        max_depth=best_params['max_depth'],
        min_samples_split=best_params['min_samples_split'],
        min_samples_leaf=best_params['min_samples_leaf'],
        random_state=42
    )
    
    # PASO 3: Hacer predicciones con ambos modelos (usando mismos parámetros)
    print("\n3.3 Realizando predicciones...")
    y_pred_rf_sklearn = predecir(rf_optimized, X_test)
    y_pred_rf_custom = predecir(rf_custom_optimized, X_test)
    
    # PASO 4: Comparar los resultados
    comparar_modelos(y_test, y_pred_rf_custom, y_pred_rf_sklearn, "Random Forest Optimizado")

    # --------------------------
    # Comparación con versión NO optimizada
    # --------------------------
    print("\n" + "="*60)
    print("4. COMPARACIÓN: OPTIMIZADO VS NO OPTIMIZADO")
    print("="*60)
    
    # Entrenar Random Forest sin optimización (parámetros por defecto)
    rf_default = random_forest_sklearn(X_train, y_train)
    y_pred_rf_default = predecir(rf_default, X_test)
    
    # Mostrar diferencias
    from sklearn.metrics import mean_squared_error, r2_score
    
    mse_default = mean_squared_error(y_test, y_pred_rf_default)
    mse_optimized = mean_squared_error(y_test, y_pred_rf_sklearn)
    r2_default = r2_score(y_test, y_pred_rf_default)
    r2_optimized = r2_score(y_test, y_pred_rf_sklearn)
    
    print(f"\nRandom Forest sin optimizar:")
    print(f"  MSE: {mse_default:.4f}")
    print(f"  R²:  {r2_default:.4f}")
    
    print(f"\nRandom Forest optimizado:")
    print(f"  MSE: {mse_optimized:.4f}")
    print(f"  R²:  {r2_optimized:.4f}")
    
    # Calcular mejora
    mejora_mse = ((mse_default - mse_optimized) / mse_default) * 100
    mejora_r2 = ((r2_optimized - r2_default) / abs(r2_default)) * 100
    
    print(f"\nMEJORA CON OPTIMIZACIÓN:")
    print(f"  MSE: {mejora_mse:+.2f}%")
    print(f"  R²:  {mejora_r2:+.2f}%")

    # --------------------------
    # PRUEBA DE REPRODUCIBILIDAD
    # --------------------------
    # print("\n" + "="*60)
    # print("5. VERIFICACIÓN DE REPRODUCIBILIDAD")
    # print("="*60)
    
    # print("\n5.1 Ejecutando el mismo experimento 3 veces...")
    
    # resultados_reproducibilidad = []
    
    # for i in range(3):
    #     print(f"\nEjecución {i+1}:")
        
    #     # Re-entrenar con los mismos parámetros y semilla
    #     rf_test = random_forest_custom(
    #         X_train, y_train, 
    #         n_estimators=best_params['n_estimators'],
    #         max_depth=best_params['max_depth'],
    #         min_samples_split=best_params['min_samples_split'],
    #         min_samples_leaf=best_params['min_samples_leaf'],
    #         random_state=42  # Misma semilla = mismo resultado
    #     )
        
    #     y_pred_test = predecir(rf_test, X_test)
    #     r2_test = r2_score(y_test, y_pred_test)
    #     mse_test = mean_squared_error(y_test, y_pred_test)
        
    #     resultados_reproducibilidad.append((r2_test, mse_test))
    #     print(f"  R²: {r2_test:.6f}, MSE: {mse_test:.6f}")
    
    # # Verificar que todos los resultados son idénticos
    # r2_values = [r[0] for r in resultados_reproducibilidad]
    # mse_values = [r[1] for r in resultados_reproducibilidad]
    
    # r2_std = np.std(r2_values)
    # mse_std = np.std(mse_values)
    
    # print(f"\n5.2 Verificación de reproducibilidad:")
    # print(f"  Desviación estándar R²: {r2_std:.10f}")
    # print(f"  Desviación estándar MSE: {mse_std:.10f}")
    
    # if r2_std < 1e-10 and mse_std < 1e-10:
    #     print("✅ EXPERIMENTO PERFECTAMENTE REPRODUCIBLE")
    # else:
    #     print("⚠️  Hay variabilidad en los resultados")

    # --------------------------
    # RESUMEN FINAL
    # --------------------------
    print("\n" + "="*60)
    print("6. RESUMEN FINAL")
    print("="*60)
    
    print(f"\nMEJORES HIPERPARÁMETROS ENCONTRADOS:")
    for param, valor in best_params.items():
        print(f"  {param}: {valor}")
    
    print(f"\nRENDIMIENTO FINAL:")
    print(f"  Random Forest sklearn optimizado - R²: {r2_score(y_test, y_pred_rf_sklearn):.4f}")
    print(f"  Random Forest manual optimizado  - R²: {r2_score(y_test, y_pred_rf_custom):.4f}")
    
    diferencia_r2 = abs(r2_score(y_test, y_pred_rf_sklearn) - r2_score(y_test, y_pred_rf_custom))
    print(f"  Diferencia entre implementaciones: {diferencia_r2:.4f}")


if __name__ == "__main__":
     
    main()