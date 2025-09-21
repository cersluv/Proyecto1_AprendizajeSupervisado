#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main Script - Problema 1: Predicción de Ingresos (>50K)
Comparación entre implementaciones propias vs sklearn con hiperparámetros optimizados
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar nuestros módulos
from preprocessing import preparar_datos
from model import (
    # Nueva función optimizada
    get_optimized_custom_models,
    
    # Modelos básicos (para comparación)
    logistic_regression_custom,
    decision_tree_custom,
    logistic_regression_sklearn,
    decision_tree_sklearn,
    
    # Funciones auxiliares
    predecir,
    evaluar_modelo
)

def imprimir_separador(titulo):
    """Imprime un separador visual con título"""
    print(f"\n{'='*60}")
    print(f" {titulo}")
    print(f"{'='*60}")

def comparar_modelos(y_test, predicciones, nombres_modelos):
    """Compara métricas de múltiples modelos"""
    resultados = []
    
    for pred, nombre in zip(predicciones, nombres_modelos):
        metricas = evaluar_modelo(y_test, pred)
        metricas['modelo'] = nombre
        resultados.append(metricas)
    
    # Crear DataFrame para mejor visualización
    df_resultados = pd.DataFrame(resultados)
    df_resultados = df_resultados[['modelo', 'accuracy', 'precision', 'recall', 'f1_score']]
    
    return df_resultados

def crear_tabla_comparacion(df_resultados):
    """Crea una tabla formateada de resultados"""
    print("\n📊 TABLA DE COMPARACIÓN DE MODELOS:")
    print("-" * 80)
    print(f"{'Modelo':<30} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for _, row in df_resultados.iterrows():
        print(f"{row['modelo']:<30} {row['accuracy']:<12.4f} {row['precision']:<12.4f} {row['recall']:<12.4f} {row['f1_score']:<12.4f}")
    
    print("-" * 80)
    
    # Encontrar el mejor modelo por cada métrica
    mejor_accuracy = df_resultados.loc[df_resultados['accuracy'].idxmax()]
    mejor_f1 = df_resultados.loc[df_resultados['f1_score'].idxmax()]
    
    print(f"\n🏆 MEJORES MODELOS:")
    print(f"   • Mejor Accuracy: {mejor_accuracy['modelo']} ({mejor_accuracy['accuracy']:.4f})")
    print(f"   • Mejor F1-Score: {mejor_f1['modelo']} ({mejor_f1['f1_score']:.4f})")

def main():
    """Función principal del análisis"""
    
    print("🚀 INICIANDO ANÁLISIS - PROBLEMA 1: PREDICCIÓN DE INGRESOS")
    print(f"⏰ Fecha y hora: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # ================================
    # 1. PREPARAR DATOS
    # ================================
    imprimir_separador("1. PREPARACIÓN DE DATOS")
    
    print("📂 Cargando y procesando dataset...")
    X_train, X_test, y_train, y_test = preparar_datos()
    
    print(f"✅ Datos preparados exitosamente!")
    print(f"   • Conjunto de entrenamiento: {X_train.shape[0]} muestras, {X_train.shape[1]} características")
    print(f"   • Conjunto de prueba: {X_test.shape[0]} muestras")
    print(f"   • Distribución de clases en entrenamiento:")
    print(f"     - Clase 0 (≤50K): {np.sum(y_train == 0)} ({np.mean(y_train == 0)*100:.1f}%)")
    print(f"     - Clase 1 (>50K): {np.sum(y_train == 1)} ({np.mean(y_train == 1)*100:.1f}%)")
    
    # ================================
    # 2. ENTRENAR MODELOS BÁSICOS (SIN OPTIMIZACIÓN)
    # ================================
    imprimir_separador("2. MODELOS CON PARÁMETROS DEFAULT")
    
    print("🔧 Entrenando modelos con parámetros por defecto...")
    
    # Modelos custom básicos
    log_custom_basic = logistic_regression_custom(X_train, y_train)
    tree_custom_basic = decision_tree_custom(X_train, y_train)
    
    # Modelos sklearn básicos
    log_sklearn_basic = logistic_regression_sklearn(X_train, y_train)
    tree_sklearn_basic = decision_tree_sklearn(X_train, y_train)
    
    print("✅ Modelos básicos entrenados!")
    
    # ================================
    # 3. OPTIMIZAR MODELOS (MÉTODO EFICIENTE)
    # ================================
    imprimir_separador("3. OPTIMIZACIÓN DE HIPERPARÁMETROS")
    
    print("🎯 Método inteligente: sklearn encuentra los mejores parámetros, custom los usa...")
    
    # Esta función optimiza sklearn primero, luego usa esos parámetros para custom
    log_custom_opt, tree_custom_opt, log_sklearn_opt, tree_sklearn_opt = get_optimized_custom_models(X_train, y_train)
    
    print("\n✅ ¡Optimización completa!")
    
    # ================================
    # 4. HACER PREDICCIONES EN TEST SET
    # ================================
    imprimir_separador("4. EVALUACIÓN EN CONJUNTO DE PRUEBA")
    
    print("🔮 Realizando predicciones en conjunto de prueba...")
    
    # Predicciones modelos básicos
    pred_log_custom_basic = predecir(log_custom_basic, X_test)
    pred_tree_custom_basic = predecir(tree_custom_basic, X_test)
    pred_log_sklearn_basic = predecir(log_sklearn_basic, X_test)
    pred_tree_sklearn_basic = predecir(tree_sklearn_basic, X_test)
    
    # Predicciones modelos optimizados
    pred_log_custom_opt = predecir(log_custom_opt, X_test)
    pred_tree_custom_opt = predecir(tree_custom_opt, X_test)
    pred_log_sklearn_opt = predecir(log_sklearn_opt, X_test)
    pred_tree_sklearn_opt = predecir(tree_sklearn_opt, X_test)
    
    # ================================
    # 5. COMPARAR RESULTADOS
    # ================================
    imprimir_separador("5. COMPARACIÓN DE RESULTADOS")
    
    # Comparar modelos básicos
    print("\n📈 COMPARACIÓN: MODELOS CON PARÁMETROS DEFAULT")
    predicciones_basic = [pred_log_custom_basic, pred_tree_custom_basic, 
                         pred_log_sklearn_basic, pred_tree_sklearn_basic]
    nombres_basic = ["Logistic Custom (Default)", "Tree Custom (Default)",
                    "Logistic Sklearn (Default)", "Tree Sklearn (Default)"]
    
    df_basic = comparar_modelos(y_test, predicciones_basic, nombres_basic)
    crear_tabla_comparacion(df_basic)
    
    # Comparar modelos optimizados
    print("\n\n📈 COMPARACIÓN: MODELOS CON HIPERPARÁMETROS OPTIMIZADOS")
    predicciones_opt = [pred_log_custom_opt, pred_tree_custom_opt,
                       pred_log_sklearn_opt, pred_tree_sklearn_opt]
    nombres_opt = ["Logistic Custom (Optimizado)", "Tree Custom (Optimizado)",
                  "Logistic Sklearn (Optimizado)", "Tree Sklearn (Optimizado)"]
    
    df_opt = comparar_modelos(y_test, predicciones_opt, nombres_opt)
    crear_tabla_comparacion(df_opt)
    
    # ================================
    # 6. ANÁLISIS FINAL
    # ================================
    imprimir_separador("6. ANÁLISIS FINAL Y CONCLUSIONES")
    
    # Combinar todos los resultados
    df_todos = pd.concat([df_basic, df_opt], ignore_index=True)
    
    print("\n🎯 ANÁLISIS COMPARATIVO:")
    
    # Mejora por optimización
    mejora_log_custom = df_opt.loc[0, 'accuracy'] - df_basic.loc[0, 'accuracy']
    mejora_tree_custom = df_opt.loc[1, 'accuracy'] - df_basic.loc[1, 'accuracy']
    mejora_log_sklearn = df_opt.loc[2, 'accuracy'] - df_basic.loc[2, 'accuracy']
    mejora_tree_sklearn = df_opt.loc[3, 'accuracy'] - df_basic.loc[3, 'accuracy']
    
    print(f"\n📊 MEJORA POR OPTIMIZACIÓN DE HIPERPARÁMETROS:")
    print(f"   • Logistic Custom: {mejora_log_custom:+.4f} ({mejora_log_custom*100:+.2f}%)")
    print(f"   • Tree Custom: {mejora_tree_custom:+.4f} ({mejora_tree_custom*100:+.2f}%)")
    print(f"   • Logistic Sklearn: {mejora_log_sklearn:+.4f} ({mejora_log_sklearn*100:+.2f}%)")
    print(f"   • Tree Sklearn: {mejora_tree_sklearn:+.4f} ({mejora_tree_sklearn*100:+.2f}%)")
    
    # Modelo ganador absoluto
    mejor_modelo_idx = df_todos['f1_score'].idxmax()
    mejor_modelo = df_todos.loc[mejor_modelo_idx]
    
    print(f"\n🏆 MODELO GANADOR ABSOLUTO:")
    print(f"   • {mejor_modelo['modelo']}")
    print(f"   • F1-Score: {mejor_modelo['f1_score']:.4f}")
    print(f"   • Accuracy: {mejor_modelo['accuracy']:.4f}")
    
    # Comparación Custom vs Sklearn
    avg_custom_opt = df_opt[df_opt['modelo'].str.contains('Custom')]['f1_score'].mean()
    avg_sklearn_opt = df_opt[df_opt['modelo'].str.contains('Sklearn')]['f1_score'].mean()
    
    print(f"\n⚔️  CUSTOM vs SKLEARN (modelos optimizados):")
    print(f"   • Promedio F1-Score Custom: {avg_custom_opt:.4f}")
    print(f"   • Promedio F1-Score Sklearn: {avg_sklearn_opt:.4f}")
    
    if avg_custom_opt > avg_sklearn_opt:
        print(f"   • 🥇 Los modelos CUSTOM superan a sklearn por {avg_custom_opt - avg_sklearn_opt:.4f} puntos!")
    else:
        print(f"   • 🥇 Los modelos SKLEARN superan a custom por {avg_sklearn_opt - avg_custom_opt:.4f} puntos!")
    
    # ================================
    # 7. GUARDAR RESULTADOS
    # ================================
    print(f"\n💾 Guardando resultados...")
    
    # Crear carpeta de resultados si no existe
    import os
    os.makedirs("resultados", exist_ok=True)
    
    # Guardar tabla de resultados
    df_todos.to_csv("resultados/comparacion_modelos_problema1.csv", index=False)
    print("   ✅ Resultados guardados en 'resultados/comparacion_modelos_problema1.csv'")
    
    print(f"\n🎉 ¡ANÁLISIS COMPLETADO EXITOSAMENTE!")
    print(f"⏰ Tiempo finalizado: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    return df_todos

if __name__ == "__main__":
    # Configurar matplotlib para evitar problemas en algunos entornos
    plt.ion()  # Modo interactivo
    
    try:
        resultados = main()
        print("\n" + "="*60)
        print(" 🚀 EJECUCIÓN COMPLETADA - Revisa los resultados arriba! 🚀")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ ERROR durante la ejecución: {str(e)}")
        print("📝 Verifica que todos los archivos estén en su lugar:")
        print("   • preprocessing.py")
        print("   • model.py") 
        print("   • Dataset en: Problema_I/dataset/IngresosPromedioAnual.xlsx")
        raise