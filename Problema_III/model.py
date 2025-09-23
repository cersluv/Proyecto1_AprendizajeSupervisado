"""
# model.py - Problema 3

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

# --------------------------
# Implementación propia
# --------------------------

def knn_custom(X_train, y_train, k=3):
    # TODO: Implementar KNN manual
    pass

def random_forest_custom(X_train, y_train):
    # TODO: Implementar Random Forest manual (simplificado)
    pass

# --------------------------
# Implementación con sklearn
# --------------------------

def knn_sklearn(X_train, y_train, k=3):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    return model

def random_forest_sklearn(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

# --------------------------
# Predicción
# --------------------------

def predecir(model, X_test):
    return model.predict(X_test)
    
"""



import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Rutas de los archivos
train_path = "Problema_III/dataset/Train.csv"
test_path = "Problema_III/dataset/Test.csv"

def load_data(path):
    """Carga el dataset desde un CSV y separa etiquetas e imágenes."""
    df = pd.read_csv(path)
    labels = df.iloc[:, 0].values
    images = df.iloc[:, 1:].values
    return images, labels

def visualize_samples(images, labels, num_samples=10):
    """Muestra las primeras 'num_samples' imágenes con sus etiquetas."""
    plt.figure(figsize=(10, 5))
    for i in range(num_samples):
        img = images[i].reshape(28, 28)  # Convertir a matriz 28x28
        plt.subplot(2, num_samples//2, i+1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Label: {labels[i]}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Cargar datos de entrenamiento
    train_images, train_labels = load_data(train_path)

    # Visualizar primeras 10 imágenes
    visualize_samples(train_images, train_labels)
