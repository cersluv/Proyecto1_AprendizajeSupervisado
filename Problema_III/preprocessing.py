import pandas as pd
import numpy as np
import cv2

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

def preprocess_and_save():
    # Cargar datos
    X_train, y_train = load_data(train_path)
    X_test, y_test = load_data(test_path)

    # Binarización
    X_train_bin = np.array([binarize_image(img.reshape(28,28)).flatten() for img in X_train])
    X_test_bin = np.array([binarize_image(img.reshape(28,28)).flatten() for img in X_test])

    # CLAHE
    X_train_clahe = np.array([apply_clahe(img.reshape(28,28)).flatten() for img in X_train])
    X_test_clahe = np.array([apply_clahe(img.reshape(28,28)).flatten() for img in X_test])

    # Normalizar a [0,1]
    X_train_bin = X_train_bin / 255.0
    X_test_bin = X_test_bin / 255.0
    X_train_clahe = X_train_clahe / 255.0
    X_test_clahe = X_test_clahe / 255.0

    # Guardar
    np.save("train_bin.npy", X_train_bin)
    np.save("test_bin.npy", X_test_bin)
    np.save("train_clahe.npy", X_train_clahe)
    np.save("test_clahe.npy", X_test_clahe)
    np.save("train_labels.npy", y_train)
    np.save("test_labels.npy", y_test)

if __name__ == "__main__":
    preprocess_and_save()
    print("Preprocesamiento completado y datos guardados.")
