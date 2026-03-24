# =============================================================================
# augment_data.py — Aumentación del dataset: multiplica x5 las imágenes
#
# CÓMO USAR:
#   1. Asegúrate de haber corrido collect_data.py y tener imágenes en dataset/raw/
#   2. Ejecuta: python augment_data.py
#   3. Las imágenes aumentadas se guardan en dataset/augmented/
#
# QUÉ HACE:
#   Por cada imagen original en dataset/raw/LETRA/, genera 5 variantes
#   aplicando transformaciones pequeñas que simulan la variabilidad natural
#   del usuario: rotaciones ligeras, desplazamientos, zoom y brillo.
#
# RESULTADO:
#   10 originales x 5 variantes = 50 imágenes por letra
#   50 x 27 letras = 1,350 imágenes totales para entrenar el SVM
# =============================================================================

import cv2
import numpy as np
import os
import random
from config import (
    ALFABETO,
    CARPETA_SEGURA,
    RUTA_DATASET_RAW,
    RUTA_DATASET_AUMENTADO,
    CANVAS_SIZE,
    FACTOR_AUMENTACION
)


# =============================================================================
# TRANSFORMACIONES DE AUMENTACIÓN
# =============================================================================

def rotar(imagen, angulo_max=15):
    """
    Rota la imagen un ángulo aleatorio dentro de [-angulo_max, +angulo_max].

    Por qué este rango:
        Nadie dibuja perfectamente vertical. Un rango de ±15 grados
        simula la inclinación natural al dibujar en el aire.
        Más de 15 grados empezaría a distorsionar la letra demasiado
        y podría confundir, por ejemplo, una 'Z' con una 'S'.

    Args:
        imagen: numpy array (H, W) en escala de grises
        angulo_max: límite de rotación en grados

    Returns:
        imagen rotada del mismo tamaño
    """
    angulo = random.uniform(-angulo_max, angulo_max)
    h, w = imagen.shape
    centro = (w // 2, h // 2)
    matriz = cv2.getRotationMatrix2D(centro, angulo, 1.0)
    return cv2.warpAffine(imagen, matriz, (w, h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def desplazar(imagen, max_px=6):
    """
    Desplaza la imagen aleatoriamente en x e y dentro de [-max_px, +max_px].

    Por qué este rango:
        El marcador no siempre queda perfectamente centrado en el canvas.
        Un desplazamiento de hasta 6 píxeles (en un canvas de 64x64)
        equivale a menos del 10% del tamaño total, suficiente para
        simular variabilidad sin deformar la letra.

    Args:
        imagen: numpy array (H, W) en escala de grises
        max_px: máximo desplazamiento en píxeles

    Returns:
        imagen desplazada del mismo tamaño
    """
    dx = random.randint(-max_px, max_px)
    dy = random.randint(-max_px, max_px)
    h, w = imagen.shape
    matriz = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(imagen, matriz, (w, h),
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def escalar(imagen, factor_min=0.85, factor_max=1.15):
    """
    Escala la imagen a un tamaño aleatorio y la devuelve al tamaño original.

    El zoom in/out simula que el usuario dibujó la letra más grande o
    más pequeña que de costumbre. Después de escalar, se recorta o
    rellena para mantener el tamaño fijo de CANVAS_SIZE x CANVAS_SIZE.

    Args:
        imagen: numpy array (H, W) en escala de grises
        factor_min: factor de escala mínimo (0.85 = 15% más pequeño)
        factor_max: factor de escala máximo (1.15 = 15% más grande)

    Returns:
        imagen escalada y reencuadrada al tamaño original
    """
    factor = random.uniform(factor_min, factor_max)
    h, w = imagen.shape
    nuevo_h = int(h * factor)
    nuevo_w = int(w * factor)

    imagen_escalada = cv2.resize(imagen, (nuevo_w, nuevo_h))
    canvas_nuevo = np.zeros((h, w), dtype=np.uint8)

    if factor >= 1.0:
        # Zoom in: recortamos el centro
        y_start = (nuevo_h - h) // 2
        x_start = (nuevo_w - w) // 2
        canvas_nuevo = imagen_escalada[y_start:y_start+h, x_start:x_start+w]
    else:
        # Zoom out: centramos la imagen más pequeña en el canvas
        y_start = (h - nuevo_h) // 2
        x_start = (w - nuevo_w) // 2
        canvas_nuevo[y_start:y_start+nuevo_h, x_start:x_start+nuevo_w] = imagen_escalada

    return canvas_nuevo


def ajustar_brillo(imagen, factor_min=0.7, factor_max=1.3):
    """
    Multiplica el brillo de cada píxel por un factor aleatorio.

    Simula variaciones de iluminación entre sesiones de recolección.
    Un factor < 1 oscurece la imagen (menos luz), > 1 la aclara.
    np.clip asegura que los valores se mantengan en [0, 255].

    Args:
        imagen: numpy array (H, W) uint8
        factor_min: mínimo multiplicador de brillo
        factor_max: máximo multiplicador de brillo

    Returns:
        imagen con brillo ajustado, mismo dtype y tamaño
    """
    factor = random.uniform(factor_min, factor_max)
    imagen_float = imagen.astype(np.float32) * factor
    return np.clip(imagen_float, 0, 255).astype(np.uint8)


def agregar_ruido(imagen, intensidad=15):
    """
    Agrega ruido gaussiano a la imagen.

    El ruido gaussiano simula el ruido electrónico natural de la cámara
    y pequeñas imperfecciones del marcador. Hace el modelo más robusto
    a condiciones no ideales de captura.

    Args:
        imagen: numpy array (H, W) uint8
        intensidad: desviación estándar del ruido (mayor = más ruido)

    Returns:
        imagen con ruido agregado
    """
    ruido = np.random.normal(0, intensidad, imagen.shape).astype(np.float32)
    imagen_ruidosa = imagen.astype(np.float32) + ruido
    return np.clip(imagen_ruidosa, 0, 255).astype(np.uint8)


def aumentar_imagen(imagen):
    """
    Aplica una combinación aleatoria de transformaciones a una imagen.

    Cada transformación se aplica con cierta probabilidad (no siempre
    todas juntas) para generar variantes más diversas. Esto evita que
    todas las variantes sean casi idénticas entre sí.

    Args:
        imagen: numpy array (H, W) uint8, imagen original

    Returns:
        imagen_aumentada: numpy array (H, W) uint8 con transformaciones aplicadas
    """
    img = imagen.copy()

    # Rotación: siempre se aplica (es la más importante)
    img = rotar(img, angulo_max=12)

    # Desplazamiento: 80% de probabilidad
    if random.random() < 0.8:
        img = desplazar(img, max_px=5)

    # Escalado: 70% de probabilidad
    if random.random() < 0.7:
        img = escalar(img)

    # Brillo: 60% de probabilidad
    if random.random() < 0.6:
        img = ajustar_brillo(img)

    # Ruido: 40% de probabilidad (solo un toque leve)
    if random.random() < 0.4:
        img = agregar_ruido(img, intensidad=8)

    return img


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def _guardar_seguro(imagen, ruta):
    """Guarda una imagen usando imencode+open para compatibilidad Unicode en Windows."""
    exito, buffer = cv2.imencode('.png', imagen)
    if exito:
        with open(ruta, 'wb') as f:
            f.write(buffer.tobytes())


def crear_carpetas_aumentadas():
    """Crea la estructura de carpetas para el dataset aumentado usando nombres ASCII seguros."""
    for letra in ALFABETO:
        nombre_carpeta = CARPETA_SEGURA[letra]
        ruta = os.path.join(RUTA_DATASET_AUMENTADO, nombre_carpeta)
        os.makedirs(ruta, exist_ok=True)
    print(f"[OK] Carpetas de aumentación listas en: {RUTA_DATASET_AUMENTADO}/")


def main():
    print("=" * 60)
    print("  AUMENTACIÓN DEL DATASET")
    print("=" * 60)

    crear_carpetas_aumentadas()

    total_generadas = 0
    total_originales = 0

    for letra in ALFABETO:
        nombre_carpeta = CARPETA_SEGURA[letra]
        ruta_raw = os.path.join(RUTA_DATASET_RAW, nombre_carpeta)

        if not os.path.exists(ruta_raw):
            print(f"  [SALTAR] {letra}: carpeta no encontrada.")
            continue

        imagenes = [f for f in os.listdir(ruta_raw) if f.endswith('.png')]

        if not imagenes:
            print(f"  [SALTAR] {letra}: no hay imágenes en {ruta_raw}")
            continue

        print(f"  Procesando letra '{letra}': {len(imagenes)} originales", end="")

        contador_letra = 0
        ruta_dest = os.path.join(RUTA_DATASET_AUMENTADO, nombre_carpeta)

        for nombre_img in imagenes:
            ruta_img = os.path.join(ruta_raw, nombre_img)
            imagen_original = cv2.imread(ruta_img, cv2.IMREAD_GRAYSCALE)

            if imagen_original is None:
                continue

            # Copiar original al dataset aumentado
            nombre_original = f"orig_{nombre_img}"
            _guardar_seguro(imagen_original,
                            os.path.join(ruta_dest, nombre_original))
            contador_letra += 1

            # Generar FACTOR_AUMENTACION variantes
            for i in range(FACTOR_AUMENTACION):
                imagen_aumentada = aumentar_imagen(imagen_original)
                nombre_aumentada = f"aug_{nombre_img.replace('.png', '')}_{i+1:02d}.png"
                _guardar_seguro(imagen_aumentada,
                                os.path.join(ruta_dest, nombre_aumentada))
                contador_letra += 1

        print(f" → {contador_letra} imágenes generadas")
        total_generadas += contador_letra
        total_originales += len(imagenes)

    print("\n" + "=" * 60)
    print(f"  Originales procesadas: {total_originales}")
    print(f"  Total generadas:       {total_generadas}")
    print(f"  Dataset en:            {RUTA_DATASET_AUMENTADO}/")
    print("  Siguiente paso: ejecuta python train_model.py")
    print("=" * 60)


if __name__ == "__main__":
    main()
