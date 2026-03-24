# =============================================================================
# classifier.py — Inferencia: convierte una trayectoria en una letra predicha
#
# Responsabilidad única: cargar el modelo entrenado y exponer una función
# limpia que recibe una trayectoria y devuelve una letra con su confianza.
#
# Este script es el puente entre la visión artificial (tracker.py) y
# el machine learning (modelo_svm.pkl). main.py solo necesita llamar
# a predecir(trayectoria) sin preocuparse de cómo funciona internamente.
# =============================================================================

import numpy as np
import joblib
import os
from canvas_utils import trayectoria_a_vector
from config import RUTA_MODELO


class Classifier:
    """
    Encapsula el modelo SVM y expone una interfaz simple para clasificar letras.

    Por qué una clase y no una función suelta:
        El modelo necesita cargarse desde disco una sola vez al inicio
        del programa (operación costosa). Una clase permite cargar en
        el constructor y reutilizar el modelo en memoria en cada llamada
        a predecir(), sin leer el archivo de disco repetidamente.

    Uso en main.py:
        clf = Classifier()                    # carga el modelo una vez
        letra, confianza = clf.predecir(trayectoria)
        if confianza > 0.6:
            word_builder.agregar_letra(letra)
    """

    def __init__(self):
        """
        Carga el modelo desde disco al inicializar.

        Lanza FileNotFoundError si el modelo no existe todavía,
        con un mensaje claro que indica qué hacer.
        """
        if not os.path.exists(RUTA_MODELO):
            raise FileNotFoundError(
                f"No se encontró el modelo en: {RUTA_MODELO}\n"
                f"Ejecuta primero: python train_model.py"
            )

        print(f"[Classifier] Cargando modelo desde: {RUTA_MODELO}")
        paquete = joblib.load(RUTA_MODELO)

        self.modelo   = paquete['modelo']
        self.encoder  = paquete['encoder']
        self.clases   = paquete['clases']
        self.precision_entrenamiento = paquete.get('precision', 0.0)

        print(f"[Classifier] Modelo cargado. Clases: {self.clases}")
        print(f"[Classifier] Precisión en entrenamiento: "
              f"{self.precision_entrenamiento * 100:.1f}%\n")

    # -------------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # -------------------------------------------------------------------------

    def predecir(self, trayectoria):
        """
        Clasifica una trayectoria y devuelve la letra más probable.

        Pipeline interno:
        1. Convierte la trayectoria a un vector de 4096 números
           usando canvas_utils.trayectoria_a_vector()
        2. Le pide al SVM que prediga la clase (número)
        3. Obtiene las probabilidades de todas las clases
        4. Convierte el número de clase de vuelta a letra con el encoder
        5. Devuelve la letra y la confianza (probabilidad de la clase ganadora)

        Args:
            trayectoria: lista de tuplas (x, y) del tracker

        Returns:
            tupla (letra, confianza):
                letra: string con la letra predicha, ej: 'A'
                confianza: float entre 0.0 y 1.0 indicando qué tan seguro
                           está el modelo. 1.0 = 100% seguro, 0.5 = adivinando

            Si la trayectoria es inválida, devuelve ('?', 0.0)
        """
        if len(trayectoria) < 5:
            return ('?', 0.0)

        # ── Convertir trayectoria a vector ───────────────────────────────────
        vector = trayectoria_a_vector(trayectoria)

        if vector.sum() == 0:
            # El canvas quedó vacío: trayectoria inválida
            return ('?', 0.0)

        # El SVM espera una matriz 2D (n_muestras, n_características)
        # Nuestra trayectoria es una sola muestra, así que la envolvemos
        # en un array de shape (1, 4096)
        X = vector.reshape(1, -1)

        # ── Predicción ───────────────────────────────────────────────────────
        clase_predicha = self.modelo.predict(X)[0]

        # Probabilidades de cada clase (requiere probability=True en el SVM)
        probabilidades = self.modelo.predict_proba(X)[0]
        confianza = float(probabilidades.max())

        # ── Convertir índice de clase a letra ────────────────────────────────
        letra = self.encoder.inverse_transform([clase_predicha])[0]

        return (letra, confianza)

    def predecir_top3(self, trayectoria):
        """
        Devuelve las 3 letras más probables con sus confianzas.

        Útil para mostrar en la UI las alternativas cuando el modelo
        no está muy seguro de su predicción principal.

        Args:
            trayectoria: lista de tuplas (x, y) del tracker

        Returns:
            lista de 3 tuplas [(letra1, conf1), (letra2, conf2), (letra3, conf3)]
            ordenadas de mayor a menor confianza.
            Devuelve lista vacía si la trayectoria es inválida.
        """
        if len(trayectoria) < 5:
            return []

        vector = trayectoria_a_vector(trayectoria)

        if vector.sum() == 0:
            return []

        X = vector.reshape(1, -1)
        probabilidades = self.modelo.predict_proba(X)[0]

        # Índices de las 3 clases con mayor probabilidad
        top3_indices = np.argsort(probabilidades)[::-1][:3]

        resultado = []
        for idx in top3_indices:
            letra = self.encoder.inverse_transform([idx])[0]
            confianza = float(probabilidades[idx])
            resultado.append((letra, confianza))

        return resultado
