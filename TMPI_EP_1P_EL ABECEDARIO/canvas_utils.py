# =============================================================================
# canvas_utils.py — Conversión de trayectorias a imágenes para el SVM
#
# Responsabilidad única: transformar una lista de puntos (x, y) en un
# vector numérico normalizado que el SVM pueda clasificar.
#
# El problema que resuelve este módulo:
#   El SVM necesita que todos sus inputs tengan exactamente el mismo tamaño.
#   Una trayectoria puede tener 20 puntos o 200 puntos, dependiendo de
#   la velocidad con que dibujó el usuario. Este módulo convierte cualquier
#   trayectoria, sin importar su tamaño, en una imagen de 64x64 = 4096
#   valores numéricos. Eso es lo que el SVM recibe como input.
#
# Flujo de transformación:
#   trayectoria (lista de N puntos) →
#   canvas negro 64x64 con la letra dibujada →
#   array aplanado de 4096 números entre 0.0 y 1.0
# =============================================================================

import cv2
import numpy as np
from config import CANVAS_SIZE, TRAZO_GROSOR


def trayectoria_a_canvas(trayectoria):
    """
    Renderiza una trayectoria de puntos sobre un canvas negro cuadrado.

    El proceso tiene dos pasos internos:
    1. Normalización de coordenadas: la trayectoria puede estar en cualquier
       parte del frame de la cámara (640x480). La "traducimos" para que quede
       centrada y escalada dentro del canvas de 64x64, con un margen de 4
       píxeles en cada borde para que las letras no queden cortadas.
    2. Dibujo de segmentos: conectamos cada punto con el siguiente con una
       línea blanca. No dibujamos puntos aislados porque las letras son
       trayectorias continuas, no nubes de puntos.

    Por qué normalizar las coordenadas:
       Si no normalizamos, la misma letra "A" dibujada en la esquina superior
       izquierda de la cámara y la misma "A" dibujada en el centro producirían
       canvas completamente diferentes. El SVM fallaría porque la posición
       no debería importar, solo la forma. La normalización hace que la forma
       sea invariante a la posición y al tamaño.

    Args:
        trayectoria: lista de tuplas (x, y) en coordenadas de la cámara

    Returns:
        canvas: numpy array (CANVAS_SIZE, CANVAS_SIZE) en escala de grises,
                valores uint8 (0-255), fondo negro, letra en blanco
                Devuelve un canvas vacío si la trayectoria tiene menos de 2 puntos.
    """
    canvas = np.zeros((CANVAS_SIZE, CANVAS_SIZE), dtype=np.uint8)

    if len(trayectoria) < 2:
        # No hay suficientes puntos para dibujar un segmento
        return canvas

    puntos = np.array(trayectoria, dtype=np.float32)

    # ── Paso 1: Normalización de coordenadas ─────────────────────────────────
    # Encontramos el bounding box de la trayectoria: los extremos en x e y
    x_min, y_min = puntos[:, 0].min(), puntos[:, 1].min()
    x_max, y_max = puntos[:, 0].max(), puntos[:, 1].max()

    ancho = x_max - x_min
    alto  = y_max - y_min

    # Si la trayectoria es un punto (ancho=0 o alto=0), no podemos normalizarla
    if ancho == 0 or alto == 0:
        return canvas

    # Margen interno para que la letra no toque los bordes del canvas
    margen = 4
    tamaño_util = CANVAS_SIZE - 2 * margen  # espacio disponible después del margen

    # Escala: qué tan grande es el canvas_util en relación al bounding box.
    # Usamos la misma escala en x e y (el mínimo de ambas) para preservar
    # la proporción de la letra. Si usáramos escalas diferentes, una "O"
    # circular podría verse como una "O" ovalada.
    escala = tamaño_util / max(ancho, alto)

    # Aplicamos la transformación: trasladar al origen, escalar, agregar margen
    puntos_norm = np.zeros_like(puntos)
    puntos_norm[:, 0] = (puntos[:, 0] - x_min) * escala + margen
    puntos_norm[:, 1] = (puntos[:, 1] - y_min) * escala + margen
    puntos_norm = puntos_norm.astype(np.int32)

    # ── Paso 2: Dibujar segmentos entre puntos consecutivos ──────────────────
    for i in range(len(puntos_norm) - 1):
        pt1 = tuple(puntos_norm[i])
        pt2 = tuple(puntos_norm[i + 1])
        cv2.line(canvas, pt1, pt2, color=255, thickness=TRAZO_GROSOR)

    return canvas


def canvas_a_vector(canvas):
    """
    Convierte el canvas 2D en un vector 1D normalizado para el SVM.

    El SVM de scikit-learn espera un array 1D por muestra, no una imagen 2D.
    La operación .flatten() toma la matriz de CANVAS_SIZE x CANVAS_SIZE y la
    convierte en un vector de CANVAS_SIZE² elementos, leyendo fila por fila.

    La normalización divide por 255.0 para convertir los valores de [0, 255]
    a [0.0, 1.0]. Esto es importante para el SVM con kernel RBF porque:
    - El kernel RBF calcula distancias euclidianas entre vectores
    - Si los valores son grandes (0-255), las distancias son grandes y el
      parámetro gamma tiene un efecto diferente al esperado
    - Con valores en [0, 1], gamma='scale' funciona correctamente

    Args:
        canvas: numpy array (CANVAS_SIZE, CANVAS_SIZE) uint8

    Returns:
        vector: numpy array 1D de CANVAS_SIZE² floats en [0.0, 1.0]
    """
    return canvas.flatten().astype(np.float32) / 255.0


def trayectoria_a_vector(trayectoria):
    """
    Pipeline completo: trayectoria → vector para el SVM.

    Combina trayectoria_a_canvas() y canvas_a_vector() en una sola llamada.
    Es la función que usan classify.py y collect_data.py para procesar
    una trayectoria antes de clasificarla o guardarla.

    Args:
        trayectoria: lista de tuplas (x, y) del tracker

    Returns:
        vector: numpy array 1D de 4096 floats en [0.0, 1.0]
    """
    canvas = trayectoria_a_canvas(trayectoria)
    return canvas_a_vector(canvas)


def guardar_imagen_letra(trayectoria, ruta_archivo):
    """
    Renderiza la trayectoria y guarda la imagen resultante en disco.

    Usada exclusivamente por collect_data.py para construir el dataset.

    Por qué usamos cv2.imencode() + open() en lugar de cv2.imwrite():
        cv2.imwrite() en Windows usa internamente la API de C++ de OpenCV
        que NO maneja rutas con caracteres Unicode (tildes, Ñ, espacios
        con codificación especial, etc.). Devuelve False silenciosamente
        sin lanzar ningún error, lo que hace muy difícil detectar el problema.

        La solución es separar las dos operaciones:
        1. cv2.imencode('.png', canvas) → codifica la imagen en memoria
           como bytes PNG, sin tocar el sistema de archivos.
        2. open(ruta, 'wb').write(bytes) → escribe los bytes en disco
           usando Python nativo, que sí maneja correctamente Unicode en Windows.

    Args:
        trayectoria: lista de tuplas (x, y) del tracker
        ruta_archivo: string con la ruta completa donde guardar la imagen,
                      por ejemplo: "dataset/raw/A/muestra_001.png"

    Returns:
        bool: True si se guardó correctamente, False si hubo un problema
    """
    canvas = trayectoria_a_canvas(trayectoria)

    if canvas.sum() == 0:
        print(f"  [ADVERTENCIA] Canvas vacío, imagen no guardada: {ruta_archivo}")
        return False

    # Codificar imagen en memoria como bytes PNG
    exito, buffer = cv2.imencode('.png', canvas)

    if not exito:
        print(f"  [ERROR] No se pudo codificar la imagen: {ruta_archivo}")
        return False

    # Escribir bytes en disco usando Python nativo (compatible con Unicode en Windows)
    try:
        with open(ruta_archivo, 'wb') as f:
            f.write(buffer.tobytes())
        return True
    except Exception as e:
        print(f"  [ERROR] No se pudo guardar: {ruta_archivo} → {e}")
        return False


def dibujar_trayectoria_en_frame(frame, trayectoria, color=(0, 255, 0), grosor=2):
    """
    Dibuja la trayectoria acumulada directamente sobre el frame de la cámara.

    Esta función es puramente de visualización (UI), se usa en main.py
    para que el usuario pueda ver en pantalla el trazo que está realizando
    mientras dibuja la letra.

    A diferencia de trayectoria_a_canvas(), aquí NO normalizamos las
    coordenadas: dibujamos los puntos exactamente donde el marcador estuvo
    en el frame, para que el trazo coincida visualmente con el movimiento real.

    Args:
        frame: numpy array BGR (H, W, 3), el frame de la cámara sobre el
               que se va a dibujar
        trayectoria: lista de tuplas (x, y)
        color: tupla BGR para el color de la línea (default: verde)
        grosor: grosor de la línea en píxeles (default: 2)

    Returns:
        frame: el mismo frame con la trayectoria dibujada encima
               (modifica el array en lugar, pero también lo devuelve
               por conveniencia de encadenamiento)
    """
    if len(trayectoria) < 2:
        return frame

    for i in range(len(trayectoria) - 1):
        cv2.line(frame, trayectoria[i], trayectoria[i + 1], color, grosor)

    return frame