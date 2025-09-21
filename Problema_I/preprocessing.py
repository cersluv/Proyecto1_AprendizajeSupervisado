# preprocessing.py - Problema 1

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preparar_datos(df=None):
    # --------------------------
    # 1. Cargar dataset si no se pasa como argumento
    # --------------------------
    if df is None:
        DATA_PATH = os.path.join("Problema_I", "dataset", "IngresosPromedioAnual.xlsx")
        df = pd.read_excel(DATA_PATH)

    # --------------------------
    # 2. Mostrar primeras filas completas
    # --------------------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print("\n🔍 Primeras filas del dataset:")
    #print(df.head(3))

    # --------------------------
    # 3. Reemplazar "?" por NaN
    # --------------------------
    df.replace("?", pd.NA, inplace=True)

    # --------------------------
    # 4. Eliminar filas con valores faltantes (Tener valores vacíos en ocupación, país de origen y clase de trabajo, afecta a la calidad del análisis)
    # --------------------------
    df.dropna(inplace=True)

    # --------------------------
    # 5. Codificar variable objetivo
    # --------------------------
    df["income"] = df["income"].apply(lambda x: 1 if x == ">50K" else 0)

    # --------------------------
    # 6. Separar X e y
    # --------------------------
    X = df.drop("income", axis=1)
    y = df["income"]

    # --------------------------
    # 7. Codificar variables categóricas
    # --------------------------
    X = pd.get_dummies(X, drop_first=True)
    #print(df.head(3))

    # --------------------------
    # 8. Escalar variables numéricas
    # --------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # print("\nPrimeras filas de X escalado:")
    # (print(X_scaled[:3])   )
    # print("\nDimensiones de X e y:")
    # print(f"X: {X_scaled.shape}, y: {y.shape}")

    # --------------------------
    # 9. Dividir en entrenamiento y prueba
    # --------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )


    return X_train, X_test, y_train, y_test


# preparar_datos()