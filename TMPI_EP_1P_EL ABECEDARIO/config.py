# =============================================================================
# config.py — Configuración global del sistema de visión artificial
# Todos los demás scripts importan sus constantes desde aquí.
# Si necesitas ajustar algo (iluminación, tiempos, rutas), este es el único
# archivo que debes modificar.
# =============================================================================

import numpy as np

# -----------------------------------------------------------------------------
# SECCIÓN 1: Detección del marcador verde en espacio HSV
# Los valores H (tono) van de 0 a 179 en OpenCV (no 0-360 como en teoría).
# El verde puro está alrededor de H=60. Usamos un rango amplio (35-85)
# para tolerar variaciones de iluminación sin perder el marcador.
# S y V los dejamos altos (100+) para ignorar verdes apagados o grisáceos
# del fondo, captando solo verdes vívidos como los de un marcador de pizarrón.
# -----------------------------------------------------------------------------
HSV_VERDE_LOWER = np.array([35, 100, 100], dtype=np.uint8)
HSV_VERDE_UPPER = np.array([85, 255, 255], dtype=np.uint8)

# Tamaño mínimo de contorno para ser considerado el marcador (en píxeles).
# Filtra ruido pequeño que pueda tener el mismo color que el marcador.
MIN_CONTOUR_AREA = 500

# -----------------------------------------------------------------------------
# SECCIÓN 2: Parámetros del canvas de trayectoria
# Cada letra se renderiza como una imagen cuadrada de CANVAS_SIZE x CANVAS_SIZE.
# 64x64 es suficiente resolución para distinguir letras y mantiene
# el vector de entrada del SVM en 4096 dimensiones (manejable en CPU).
# -----------------------------------------------------------------------------
CANVAS_SIZE = 64

# Grosor de la línea al dibujar la trayectoria sobre el canvas.
TRAZO_GROSOR = 3

# -----------------------------------------------------------------------------
# SECCIÓN 3: Tiempos de espera para detectar señales del usuario
# Todos los tiempos están en FRAMES, no en segundos.
# A ~30fps: 45 frames ≈ 1.5 segundos, 60 frames ≈ 2 segundos.
# Ajusta estos valores si tu cámara corre a diferente velocidad.
# -----------------------------------------------------------------------------

# Frames sin detectar marcador para interpretar "fin de letra"
FRAMES_FIN_LETRA = 45      # ~1.5 segundos de ausencia del marcador

# Frames sin movimiento para interpretar "fin de palabra"
# (marcador quieto en el mismo punto)
FRAMES_FIN_PALABRA = 60    # ~2 segundos de marcador inmóvil

# Distancia máxima en píxeles para considerar que el marcador "no se movió"
UMBRAL_MOVIMIENTO = 15

# -----------------------------------------------------------------------------
# SECCIÓN 4: Rutas del sistema de archivos
# Centralizar rutas aquí evita strings hardcodeados dispersos en el código.
# -----------------------------------------------------------------------------
RUTA_DATASET_RAW       = "dataset/raw"        # imágenes originales recolectadas
RUTA_DATASET_AUMENTADO = "dataset/augmented"  # imágenes después de augmentación
RUTA_MODELO            = "modelo_svm.pkl"     # modelo entrenado serializado
RUTA_SALIDA_TEXTO      = "palabras_detectadas.txt"  # archivo final de palabras

# -----------------------------------------------------------------------------
# SECCIÓN 5: El alfabeto español completo (27 clases)
# El orden importa: el índice de cada letra en esta lista corresponde
# al índice de clase que usará el SVM. La Ñ va entre N y O.
# -----------------------------------------------------------------------------
ALFABETO = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
    'N', 'Ñ', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
]

# Número total de clases (debe coincidir con len(ALFABETO))
NUM_CLASES = len(ALFABETO)  # 27

# -----------------------------------------------------------------------------
# SECCIÓN 6: Hiperparámetros del modelo SVM
# C controla el balance entre margen amplio y errores de clasificación.
#   - C pequeño (0.1-1): más margen, acepta más errores → menos sobreajuste
#   - C grande (10-100): menos margen, tolera pocos errores → más sobreajuste
# gamma='scale' es el valor recomendado para RBF: calcula gamma automáticamente
# basándose en la varianza de los datos de entrada.
# -----------------------------------------------------------------------------
SVM_C      = 10
SVM_KERNEL = 'rbf'
SVM_GAMMA  = 'scale'

# Porcentaje de datos reservados para validación (20% del dataset aumentado)
TEST_SIZE = 0.2

# -----------------------------------------------------------------------------
# SECCIÓN 7: Cuántas muestras recolectar y cuánto aumentarlas
# -----------------------------------------------------------------------------
MUESTRAS_POR_LETRA    = 10   # imágenes que dibujarás manualmente por letra
FACTOR_AUMENTACION    = 5    # multiplicador: 10 originales → 50 por letra

# -----------------------------------------------------------------------------
# SECCIÓN 8: Parámetros de visualización en pantalla (main.py)
# -----------------------------------------------------------------------------
COLOR_TRAYECTORIA = (0, 255, 0)    # verde (BGR)
COLOR_TEXTO_UI    = (255, 255, 255) # blanco (BGR)
COLOR_ALERTA      = (0, 0, 255)    # rojo (BGR) para señal de fin de sesión
FUENTE_UI         = 1              # cv2.FONT_HERSHEY_SIMPLEX
ESCALA_FUENTE     = 0.7
GROSOR_FUENTE     = 2