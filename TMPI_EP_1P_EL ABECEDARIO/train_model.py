# =============================================================================
# train_model.py — Entrenamiento del modelo SVM para clasificación de letras
#
# CÓMO USAR:
#   1. Asegúrate de haber corrido augment_data.py primero
#   2. Ejecuta: python train_model.py
#   3. El modelo entrenado se guarda en modelo_svm.pkl
#
# QUÉ HACE:
#   1. Carga todas las imágenes del dataset aumentado
#   2. Convierte cada imagen a un vector de 4096 números
#   3. Divide en 80% entrenamiento / 20% validación
#   4. Entrena el SVM con kernel RBF
#   5. Evalúa la precisión y muestra el reporte
#   6. Guarda el modelo en disco para usarlo en main.py
# =============================================================================

import cv2
import numpy as np
import os
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from config import (
    ALFABETO,
    CARPETA_SEGURA,
    CARPETA_A_LETRA,
    RUTA_DATASET_AUMENTADO,
    RUTA_MODELO,
    SVM_C,
    SVM_KERNEL,
    SVM_GAMMA,
    TEST_SIZE,
    CANVAS_SIZE
)


# =============================================================================
# CARGA DEL DATASET
# =============================================================================

def cargar_dataset():
    """
    Lee todas las imágenes del dataset aumentado y las convierte a vectores.

    Para cada imagen:
    1. La lee en escala de grises (no necesitamos color, solo la forma del trazo)
    2. La redimensiona a CANVAS_SIZE x CANVAS_SIZE por si acaso
    3. La aplana a un vector 1D y normaliza a [0.0, 1.0]
    4. Registra su etiqueta (la letra que representa)

    Returns:
        X: numpy array (N, CANVAS_SIZE²) con los vectores de todas las imágenes
        y: numpy array (N,) con las etiquetas (letras) correspondientes
    """
    X = []  # vectores de imágenes
    y = []  # etiquetas (letras)

    print("Cargando dataset...")

    for letra in ALFABETO:
        nombre_carpeta = CARPETA_SEGURA[letra]
        ruta_letra = os.path.join(RUTA_DATASET_AUMENTADO, nombre_carpeta)

        if not os.path.exists(ruta_letra):
            print(f"  [ADVERTENCIA] No se encontró carpeta para: {letra} ({nombre_carpeta})")
            continue

        imagenes = [f for f in os.listdir(ruta_letra) if f.endswith('.png')]

        if not imagenes:
            print(f"  [ADVERTENCIA] Sin imágenes para: {letra}")
            continue

        for nombre_img in imagenes:
            ruta_img = os.path.join(ruta_letra, nombre_img)
            imagen = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)

            if imagen is None:
                continue

            if imagen.shape != (CANVAS_SIZE, CANVAS_SIZE):
                imagen = cv2.resize(imagen, (CANVAS_SIZE, CANVAS_SIZE))

            vector = imagen.flatten().astype(np.float32) / 255.0
            X.append(vector)
            y.append(letra)   # etiqueta real: 'Ñ', no 'NTILDE'

        print(f"  [OK] {letra}: {len(imagenes)} imágenes cargadas")

    X = np.array(X)
    y = np.array(y)

    print(f"\nDataset cargado: {X.shape[0]} imágenes, {X.shape[1]} características")
    print(f"Clases encontradas: {np.unique(y)}\n")

    return X, y


# =============================================================================
# ENTRENAMIENTO
# =============================================================================

def entrenar_modelo(X, y):
    """
    Entrena el SVM con los datos cargados.

    Proceso:
    1. Codifica las etiquetas de texto ('A','B'...) a números (0,1,2...)
       porque el SVM trabaja con números, no strings.
    2. Divide los datos en entrenamiento (80%) y validación (20%).
    3. Crea y entrena el SVM con los hiperparámetros de config.py.
    4. Evalúa el modelo en el conjunto de validación.
    5. Guarda el modelo y el codificador juntos en un diccionario.

    Por qué guardar el LabelEncoder junto con el modelo:
        El SVM predice números (0, 1, 2...). Para convertir esos números
        de vuelta a letras ('A', 'B', 'C'...) necesitamos el mismo
        LabelEncoder que se usó al entrenar. Si no lo guardamos,
        no podemos saber qué número corresponde a qué letra.

    Args:
        X: numpy array (N, 4096) con los vectores de imágenes
        y: numpy array (N,) con las etiquetas en texto

    Returns:
        paquete: diccionario con el modelo y el codificador guardados juntos
    """
    # ── Codificar etiquetas ──────────────────────────────────────────────────
    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)

    print(f"Clases codificadas: {list(encoder.classes_)}")
    print(f"Índices:            {list(range(len(encoder.classes_)))}\n")

    # ── Dividir dataset ──────────────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_encoded,
        test_size=TEST_SIZE,
        random_state=42,        # semilla fija para reproducibilidad
        stratify=y_encoded      # mantiene proporciones de clases en ambos sets
    )

    print(f"Entrenamiento: {len(X_train)} muestras")
    print(f"Validación:    {len(X_val)} muestras\n")

    # ── Crear y entrenar SVM ─────────────────────────────────────────────────
    print("Entrenando SVM... (puede tomar 1-3 minutos)")
    print(f"  Kernel: {SVM_KERNEL}, C: {SVM_C}, gamma: {SVM_GAMMA}\n")

    modelo = SVC(
        C=SVM_C,
        kernel=SVM_KERNEL,
        gamma=SVM_GAMMA,
        probability=True,   # habilita predict_proba() para ver confianza
        random_state=42
    )

    modelo.fit(X_train, y_train)
    print("[OK] Entrenamiento completado.\n")

    # ── Evaluar en validación ────────────────────────────────────────────────
    y_pred = modelo.predict(X_val)
    precision = accuracy_score(y_val, y_pred)

    print("=" * 60)
    print(f"  PRECISIÓN EN VALIDACIÓN: {precision * 100:.2f}%")
    print("=" * 60)

    # Reporte detallado por letra
    print("\nReporte por letra:")
    print(classification_report(
        y_val, y_pred,
        target_names=encoder.classes_
    ))

    # Advertencia si la precisión es baja
    if precision < 0.70:
        print("[ADVERTENCIA] Precisión menor al 70%.")
        print("  Considera recolectar más muestras o revisar la calidad")
        print("  de los trazos en el dataset.\n")
    elif precision < 0.85:
        print("[INFO] Precisión aceptable. Para mejorar, puedes recolectar")
        print("  más muestras de las letras con menor precision en el reporte.\n")
    else:
        print("[EXCELENTE] El modelo está listo para usar en producción.\n")

    # ── Empaquetar modelo y encoder juntos ───────────────────────────────────
    paquete = {
        'modelo'  : modelo,
        'encoder' : encoder,
        'clases'  : list(encoder.classes_),
        'precision': precision
    }

    return paquete


# =============================================================================
# GUARDAR MODELO
# =============================================================================

def guardar_modelo(paquete):
    """
    Serializa el modelo entrenado a disco usando joblib.

    joblib es más eficiente que pickle para arrays numpy grandes
    (como los vectores de soporte del SVM) porque usa compresión
    y serialización optimizada para datos numéricos.

    Args:
        paquete: diccionario con modelo, encoder y metadatos
    """
    joblib.dump(paquete, RUTA_MODELO)
    tamaño = os.path.getsize(RUTA_MODELO) / 1024  # en KB
    print(f"[GUARDADO] Modelo guardado en: {RUTA_MODELO} ({tamaño:.1f} KB)")


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    print("=" * 60)
    print("  ENTRENAMIENTO DEL MODELO SVM")
    print("=" * 60 + "\n")

    # Verificar que existe el dataset
    if not os.path.exists(RUTA_DATASET_AUMENTADO):
        print("[ERROR] No se encontró el dataset aumentado.")
        print(f"  Ruta esperada: {RUTA_DATASET_AUMENTADO}")
        print("  Ejecuta primero: python augment_data.py")
        return

    # Cargar datos
    X, y = cargar_dataset()

    if len(X) == 0:
        print("[ERROR] No se cargaron imágenes. Revisa el dataset.")
        return

    # Entrenar
    paquete = entrenar_modelo(X, y)

    # Guardar
    guardar_modelo(paquete)

    print("\n" + "=" * 60)
    print("  ¡MODELO LISTO!")
    print(f"  Precisión: {paquete['precision'] * 100:.2f}%")
    print(f"  Archivo:   {RUTA_MODELO}")
    print("  Siguiente paso: ejecuta python main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
