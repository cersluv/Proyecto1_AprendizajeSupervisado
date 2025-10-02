import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def preparar_datos(df=None):
    # --------------------------
    # Cargar dataset si no se pasa como argumento
    # --------------------------
    if df is None:
        DATA_PATH = os.path.join("dataset/CalidadAire.csv")
        print(DATA_PATH)
        df = pd.read_csv(DATA_PATH)

    # --------------------------
    # Mostrar primeras filas completas
    # --------------------------
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    
    # --------------------------
    # Eliminar columnas irrelevantes
    # --------------------------

    df = df.drop(columns=["Unnamed: 15", "Unnamed: 16"])


    # --------------------------
    # Reemplazar valores nulos por NaN
    # --------------------------
    df.replace(-200, np.nan, inplace=True)
    df = df.dropna(how='all')  #totalmente vacíos
    df = df.dropna(subset=['CO(GT)']) # con valores en CO(GT) (la variable a predecir) vacíos
    #print(len(df))
    if df['NMHC(GT)'].isna().mean() > 0.8:  # Si NMHC(GT) tiene >80% NaN, se va
        df = df.drop(columns=['NMHC(GT)'])

    # --------------------------
    # Cambiar formato de fecha y hora
    # --------------------------
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
    df['hour'] = df['datetime'].dt.hour
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['month'] = df['datetime'].dt.month

    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    df['datetime'] = df['datetime'].dt.date

    df = df.drop(columns=['Date', 'Time'])
    df.reset_index(drop=True, inplace=True)

    #print(df.head(-10))

    # --------------------------
    # interpolar valores faltantes
    # --------------------------

    df[df.select_dtypes(include=[np.number]).columns] = \
        df.select_dtypes(include=[np.number]).interpolate(method='linear', limit_direction='both')


    # --------------------------
    # Separar X e y
    # --------------------------

    y = df['CO(GT)']
    X = df.drop(columns=['CO(GT)', 'datetime'])

    # --------------------------
    # Crear bins para estratificar en regresión
    # --------------------------
    y_bins = pd.qcut(y, q=10)
    # print(bins.value_counts())
    print(y_bins.value_counts().sort_index())


    # --------------------------
    # Dividir en conjunto de entrenamiento y prueba
    # --------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y_bins
    )

    # --------------------------
    # Escalar variables numéricas para regresión lineal
    # --------------------------

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)


    #print(X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)
    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled


preparar_datos()


"""
#Ejemplo de uso

X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled = preparar_datos()

# Modelo sensible a escala → regresión lineal
lr_model = linear_regression_sklearn(X_train_scaled, y_train)
y_pred_lr = predecir(lr_model, X_test_scaled)

# Modelo no sensible a escala → random forest
rf_model = random_forest_sklearn(X_train, y_train)
y_pred_rf = predecir(rf_model, X_test)


"""
