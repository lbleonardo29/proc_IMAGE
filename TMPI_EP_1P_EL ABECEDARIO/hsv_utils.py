# =============================================================================
# hsv_utils.py — Utilidades de procesamiento de color en espacio HSV
#
# Responsabilidad única: recibir un frame BGR de la cámara y devolver
# una máscara binaria limpia donde el marcador verde = blanco (255)
# y el resto del mundo = negro (0).
#
# Por qué HSV y no BGR:
#   En BGR, la "verdad" de un color cambia con la iluminación porque
#   los 3 canales (B, G, R) se ven afectados simultáneamente por sombras
#   y luz. En HSV, el canal H (Hue/Tono) codifica el color puro y es
#   relativamente estable ante cambios de iluminación. Solo el canal V
#   (Value/Brillo) cambia con la luz, y nosotros simplemente ponemos un
#   umbral mínimo de V para ignorar zonas muy oscuras.
# =============================================================================

import cv2
import numpy as np
from config import HSV_VERDE_LOWER, HSV_VERDE_UPPER, MIN_CONTOUR_AREA


def bgr_a_hsv(frame):
    """
    Convierte un frame de BGR (formato nativo de OpenCV) a HSV.

    OpenCV captura en BGR por razones históricas (compatibilidad con la API
    de Windows de los años 90). Siempre hay que convertir explícitamente.

    Args:
        frame: numpy array (H, W, 3) en formato BGR, directo de VideoCapture

    Returns:
        numpy array (H, W, 3) en formato HSV
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


def crear_mascara_verde(frame_hsv):
    """
    Genera una máscara binaria que aísla los píxeles del color verde del marcador.

    cv2.inRange() recorre cada píxel del frame_hsv y verifica si sus tres
    valores (H, S, V) caen dentro del rango [lower, upper]. Si sí, escribe
    255 en ese píxel de la máscara. Si no, escribe 0.

    El resultado es una imagen en escala de grises (no tiene canales de color)
    donde blanco = "aquí hay marcador verde" y negro = "aquí no hay nada".

    Args:
        frame_hsv: numpy array (H, W, 3) en espacio de color HSV

    Returns:
        mascara: numpy array (H, W) binario, valores 0 o 255
    """
    mascara = cv2.inRange(frame_hsv, HSV_VERDE_LOWER, HSV_VERDE_UPPER)
    return mascara


def limpiar_mascara(mascara):
    """
    Aplica operaciones morfológicas para eliminar ruido de la máscara.

    La máscara cruda de inRange() suele tener dos tipos de imperfecciones:
    1. Píxeles blancos aislados (ruido): objetos pequeños que casualmente
       tienen el mismo tono verde que el marcador (reflejo de luz, etc.)
    2. Huecos negros dentro del marcador: la reflectividad irregular de la
       superficie del marcador puede hacer que el centro sea más brillante
       y salga fuera del rango HSV.

    Las operaciones morfológicas corrigen ambos problemas:
    - OPENING (erosión → dilatación): elimina los puntos blancos pequeños.
      Primero "encoge" todas las regiones blancas (erosión), matando los
      puntos pequeños porque desaparecen completamente. Luego "expande"
      las regiones que sobrevivieron (dilatación), devolviendo su tamaño
      original al marcador grande.
    - CLOSING (dilatación → erosión): rellena huecos negros dentro de
      regiones blancas. Primero expande el blanco hasta tapar los huecos,
      luego encoge de vuelta al tamaño original.

    El kernel es la "brocha" que usa cada operación. Un kernel de 5x5
    significa que cada píxel se evalúa en el contexto de sus 25 vecinos.

    Args:
        mascara: numpy array binario (H, W) con posible ruido

    Returns:
        mascara_limpia: numpy array binario (H, W) sin ruido
    """
    # Kernel cuadrado 5x5 para las operaciones morfológicas
    kernel = np.ones((5, 5), np.uint8)

    # Opening: elimina ruido pequeño
    mascara_abierta = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    # Closing: rellena huecos dentro del marcador
    mascara_cerrada = cv2.morphologyEx(mascara_abierta, cv2.MORPH_CLOSE, kernel)

    return mascara_cerrada


def procesar_frame(frame):
    """
    Pipeline completo: frame BGR → máscara limpia.

    Esta es la función principal del módulo. Encadena las tres operaciones
    anteriores en una sola llamada, que es lo que los demás scripts usarán.

    Flujo interno:
        frame (BGR) → frame_hsv → mascara_cruda → mascara_limpia

    Args:
        frame: numpy array BGR directo de cv2.VideoCapture.read()

    Returns:
        mascara_limpia: numpy array binario (H, W), listo para tracker.py
    """
    frame_hsv    = bgr_a_hsv(frame)
    mascara_cruda = crear_mascara_verde(frame_hsv)
    mascara_limpia = limpiar_mascara(mascara_cruda)
    return mascara_limpia


def ajustar_hsv_interactivo(frame):
    """
    Herramienta de calibración: muestra trackbars para ajustar rangos HSV
    en tiempo real. Útil para calibrar cuando cambias de iluminación o
    de marcador.

    CÓMO USAR:
        Llama a esta función con un frame de tu cámara, ajusta los sliders
        hasta que el marcador aparezca completamente blanco y el fondo
        completamente negro, y anota los valores resultantes en config.py.

    Args:
        frame: numpy array BGR de la cámara

    Returns:
        lower, upper: arrays numpy con los rangos HSV calibrados
    """
    def nada(x):
        pass  # callback vacío requerido por cv2.createTrackbar

    cv2.namedWindow("Calibrador HSV")

    # Crear 6 trackbars: mínimo y máximo para cada canal H, S, V
    cv2.createTrackbar("H min", "Calibrador HSV", 35,  179, nada)
    cv2.createTrackbar("H max", "Calibrador HSV", 85,  179, nada)
    cv2.createTrackbar("S min", "Calibrador HSV", 100, 255, nada)
    cv2.createTrackbar("S max", "Calibrador HSV", 255, 255, nada)
    cv2.createTrackbar("V min", "Calibrador HSV", 100, 255, nada)
    cv2.createTrackbar("V max", "Calibrador HSV", 255, 255, nada)

    print("Ajusta los sliders hasta que el marcador sea blanco puro.")
    print("Presiona 'q' cuando estés satisfecho con la calibración.")

    while True:
        h_min = cv2.getTrackbarPos("H min", "Calibrador HSV")
        h_max = cv2.getTrackbarPos("H max", "Calibrador HSV")
        s_min = cv2.getTrackbarPos("S min", "Calibrador HSV")
        s_max = cv2.getTrackbarPos("S max", "Calibrador HSV")
        v_min = cv2.getTrackbarPos("V min", "Calibrador HSV")
        v_max = cv2.getTrackbarPos("V max", "Calibrador HSV")

        lower = np.array([h_min, s_min, v_min], dtype=np.uint8)
        upper = np.array([h_max, s_max, v_max], dtype=np.uint8)

        frame_hsv = bgr_a_hsv(frame)
        mascara   = cv2.inRange(frame_hsv, lower, upper)

        cv2.imshow("Calibrador HSV", mascara)
        cv2.imshow("Frame original", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    print(f"\nRangos calibrados:")
    print(f"HSV_VERDE_LOWER = np.array({list(lower)}, dtype=np.uint8)")
    print(f"HSV_VERDE_UPPER = np.array({list(upper)}, dtype=np.uint8)")
    print("Copia estos valores en config.py")

    return lower, upper
