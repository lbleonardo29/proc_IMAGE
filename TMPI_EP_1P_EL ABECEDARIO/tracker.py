# =============================================================================
# tracker.py — Seguimiento del marcador y detección de señales del usuario
#
# Responsabilidad única: recibir la máscara binaria limpia de hsv_utils.py
# y mantener el estado del marcador a lo largo del tiempo:
#   - ¿Dónde está el marcador ahora? (centroide)
#   - ¿Está en movimiento o quieto?
#   - ¿Cuánto tiempo lleva desaparecido?
#   - ¿Se activó alguna señal? (fin de letra / fin de palabra / fin de sesión)
#
# Concepto clave — Máquina de estados:
#   El tracker implementa una máquina de estados simple con 3 estados:
#   DIBUJANDO → el marcador está visible y moviéndose (acumulamos trayectoria)
#   ESPERANDO → el marcador desapareció (contamos frames de ausencia)
#   QUIETO    → el marcador está visible pero inmóvil (contamos frames quieto)
# =============================================================================

import cv2
import numpy as np
from config import (
    MIN_CONTOUR_AREA,
    FRAMES_FIN_LETRA,
    FRAMES_FIN_PALABRA,
    UMBRAL_MOVIMIENTO
)

# --- Estados internos de la máquina de estados ---
ESTADO_DIBUJANDO = "DIBUJANDO"
ESTADO_ESPERANDO = "ESPERANDO"   # marcador ausente
ESTADO_QUIETO    = "QUIETO"      # marcador visible pero inmóvil


class Tracker:
    """
    Clase que mantiene el estado del marcador a lo largo del tiempo.

    Por qué una clase y no funciones sueltas:
        El tracker necesita "memoria" entre frames: cuántos frames lleva
        el marcador ausente, cuántos lleva quieto, cuál fue la última posición,
        etc. Una clase encapsula ese estado de forma limpia, evitando variables
        globales dispersas por el código.

    Uso típico:
        tracker = Tracker()
        while True:
            ret, frame = cap.read()
            mascara = procesar_frame(frame)
            resultado = tracker.actualizar(mascara)
            if resultado['señal'] == 'FIN_LETRA':
                trayectoria = tracker.obtener_trayectoria()
                tracker.resetear_trayectoria()
    """

    def __init__(self):
        """Inicializa el tracker con estado limpio."""
        self.estado               = ESTADO_ESPERANDO
        self.trayectoria          = []      # lista de (x, y) del marcador
        self.ultima_posicion      = None    # último centroide detectado
        self.frames_sin_marcador  = 0       # contador de ausencia
        self.frames_quieto        = 0       # contador de inmovilidad

    # -------------------------------------------------------------------------
    # MÉTODO PRINCIPAL
    # -------------------------------------------------------------------------

    def actualizar(self, mascara):
        """
        Procesa un frame (representado por su máscara) y actualiza el estado.

        Este método es el corazón del tracker. En cada frame hace lo siguiente:
        1. Busca el marcador en la máscara
        2. Si lo encuentra, calcula su centroide y lo agrega a la trayectoria
        3. Evalúa si el marcador está quieto o moviéndose
        4. Si no lo encuentra, incrementa el contador de ausencia
        5. Verifica si algún contador llegó al umbral para emitir una señal

        Args:
            mascara: numpy array binario (H, W) de hsv_utils.procesar_frame()

        Returns:
            dict con las siguientes claves:
                'centroide'  : (x, y) del marcador o None si no fue detectado
                'señal'      : None | 'FIN_LETRA' | 'FIN_PALABRA' | 'FIN_SESION'
                'trayectoria': copia de la trayectoria actual (lista de puntos)
                'estado'     : string con el estado actual de la máquina
        """
        centroide = self._encontrar_centroide(mascara)
        señal = None

        if centroide is not None:
            # ── El marcador ES visible en este frame ──────────────────────────
            self.frames_sin_marcador = 0  # reiniciamos el contador de ausencia

            if self.ultima_posicion is not None:
                distancia = self._distancia(centroide, self.ultima_posicion)

                if distancia < UMBRAL_MOVIMIENTO:
                    # El marcador apenas se movió → está quieto
                    self.frames_quieto += 1
                    self.estado = ESTADO_QUIETO

                    if self.frames_quieto >= FRAMES_FIN_PALABRA:
                        señal = 'FIN_PALABRA'
                        self.frames_quieto = 0
                else:
                    # El marcador se movió → está dibujando
                    self.frames_quieto = 0
                    self.estado = ESTADO_DIBUJANDO
                    self.trayectoria.append(centroide)
            else:
                # Primera vez que detectamos el marcador
                self.estado = ESTADO_DIBUJANDO
                self.trayectoria.append(centroide)

            self.ultima_posicion = centroide

        else:
            # ── El marcador NO es visible en este frame ───────────────────────
            self.frames_sin_marcador += 1
            self.frames_quieto = 0
            self.estado = ESTADO_ESPERANDO

            if self.frames_sin_marcador == FRAMES_FIN_LETRA:
                # Exactamente en el frame umbral emitimos la señal
                # (usamos == para emitirla solo una vez, no en cada frame siguiente)
                if len(self.trayectoria) > 5:
                    # Solo emitimos si hay suficientes puntos para ser una letra
                    señal = 'FIN_LETRA'

        return {
            'centroide'  : centroide,
            'señal'      : señal,
            'trayectoria': list(self.trayectoria),  # copia para evitar mutación
            'estado'     : self.estado
        }

    # -------------------------------------------------------------------------
    # MÉTODOS DE GESTIÓN DE TRAYECTORIA
    # -------------------------------------------------------------------------

    def resetear_trayectoria(self):
        """
        Limpia la trayectoria después de que una letra fue procesada.
        También resetea la última posición para que la siguiente letra
        empiece limpia sin conectarse con la anterior.
        """
        self.trayectoria     = []
        self.ultima_posicion = None
        self.frames_sin_marcador = 0
        self.frames_quieto   = 0

    def obtener_trayectoria(self):
        """
        Devuelve una copia de la trayectoria actual.
        Devolver una copia (no la referencia) es importante para evitar
        que código externo modifique accidentalmente el estado interno.

        Returns:
            lista de tuplas (x, y) representando el recorrido del marcador
        """
        return list(self.trayectoria)

    def tiene_trayectoria(self):
        """
        Indica si hay puntos acumulados en la trayectoria actual.
        Útil para saber si vale la pena procesar antes de clasificar.

        Returns:
            bool: True si hay al menos 5 puntos en la trayectoria
        """
        return len(self.trayectoria) > 5

    # -------------------------------------------------------------------------
    # MÉTODOS PRIVADOS (auxiliares internos)
    # -------------------------------------------------------------------------

    def _encontrar_centroide(self, mascara):
        """
        Busca el contorno más grande en la máscara y calcula su centroide.

        Por qué contornos y no simplemente "el píxel más brillante":
            El marcador ocupa un área, no un punto. El centroide del contorno
            más grande nos da el centro geométrico de esa área, que es más
            estable que un píxel individual (menos sensible a ruido).

        Los momentos de imagen son valores estadísticos calculados sobre la
        distribución de píxeles de un contorno. Los momentos M00, M10, M01
        son específicamente:
            M00 = área total del contorno (número de píxeles blancos)
            M10 = suma de coordenadas x de todos los píxeles blancos
            M01 = suma de coordenadas y de todos los píxeles blancos
        El centroide es simplemente M10/M00 y M01/M00 (promedio de x e y).

        Args:
            mascara: numpy array binario (H, W)

        Returns:
            (cx, cy): tupla de enteros con la posición del centroide,
                      o None si no se detectó ningún contorno válido
        """
        contornos, _ = cv2.findContours(
            mascara,
            cv2.RETR_EXTERNAL,      # solo contornos externos (sin huecos)
            cv2.CHAIN_APPROX_SIMPLE # comprime segmentos rectos para eficiencia
        )

        if not contornos:
            return None

        # Tomamos el contorno de mayor área (el marcador)
        contorno_mayor = max(contornos, key=cv2.contourArea)

        if cv2.contourArea(contorno_mayor) < MIN_CONTOUR_AREA:
            # El contorno es demasiado pequeño, probablemente es ruido
            return None

        # Calculamos el centroide usando momentos de imagen
        momentos = cv2.moments(contorno_mayor)

        if momentos["m00"] == 0:
            # División por cero: contorno degenerado, lo ignoramos
            return None

        cx = int(momentos["m10"] / momentos["m00"])
        cy = int(momentos["m01"] / momentos["m00"])

        return (cx, cy)

    @staticmethod
    def _distancia(p1, p2):
        """
        Calcula la distancia euclidiana entre dos puntos (x1,y1) y (x2,y2).

        La distancia euclidiana es la "distancia en línea recta" del teorema
        de Pitágoras: sqrt((x2-x1)² + (y2-y1)²). La usamos para determinar
        si el marcador se movió lo suficiente entre frames.

        Args:
            p1, p2: tuplas (x, y)

        Returns:
            float: distancia en píxeles entre los dos puntos
        """
        return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
