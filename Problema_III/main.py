import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    return acc, f1

if __name__ == "__main__":
    # Cargar datos
    X_train_bin = np.load("train_bin.npy")
    X_test_bin = np.load("test_bin.npy")
    X_train_clahe = np.load("train_clahe.npy")
    X_test_clahe = np.load("test_clahe.npy")
    y_train = np.load("train_labels.npy")
    y_test = np.load("test_labels.npy")

    # Modelos
    knn = KNeighborsClassifier(n_neighbors=5)
    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    print("=== BINARIZADO ===")
    acc_knn_bin, f1_knn_bin = evaluate_model(knn, X_train_bin, y_train, X_test_bin, y_test)
    acc_rf_bin, f1_rf_bin = evaluate_model(rf, X_train_bin, y_train, X_test_bin, y_test)
    print(f"KNN -> Accuracy: {acc_knn_bin:.4f}, F1 macro: {f1_knn_bin:.4f}")
    print(f"RF  -> Accuracy: {acc_rf_bin:.4f}, F1 macro: {f1_rf_bin:.4f}")

    print("\n=== CLAHE ===")
    acc_knn_clahe, f1_knn_clahe = evaluate_model(knn, X_train_clahe, y_train, X_test_clahe, y_test)
    acc_rf_clahe, f1_rf_clahe = evaluate_model(rf, X_train_clahe, y_train, X_test_clahe, y_test)
    print(f"KNN -> Accuracy: {acc_knn_clahe:.4f}, F1 macro: {f1_knn_clahe:.4f}")
    print(f"RF  -> Accuracy: {acc_rf_clahe:.4f}, F1 macro: {f1_rf_clahe:.4f}")
