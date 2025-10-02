# acá se van a colocar las funciones que se usan en varios lados, por ejemplo las accurancies y esas cosas jeje

import numpy as np

def calcular_metricas(y_true, y_pred):
    """
    Calcula Accuracy, Precision, Recall y F1-score
    usando las fórmulas matemáticas básicas.
    
    y_true : array-like de etiquetas reales (0 o 1)
    y_pred : array-like de predicciones (0 o 1)
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calcular TP, TN, FP, FN
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == 0) & (y_pred == 0))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    # Fórmulas
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score
    }
