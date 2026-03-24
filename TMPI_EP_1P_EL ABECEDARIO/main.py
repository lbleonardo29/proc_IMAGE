# =============================================================================
# main.py — Sistema completo de reconocimiento de letras en tiempo real
#
# CÓMO USAR:
#   1. Asegúrate de haber corrido en orden:
#      python collect_data.py → python augment_data.py → python train_model.py
#   2. Ejecuta: python main.py
#
# SEÑALES EN TIEMPO REAL:
#   - Marcador ausente ~1.5s  → reconoce la letra dibujada
#   - Marcador quieto  ~2s    → fin de palabra (agrega espacio)
#   - Tecla 'q' o mostrar rojo ~2s → fin de sesión (guarda .txt y cierra)
#
# ESTE SCRIPT NO HACE PROCESAMIENTO PROPIO:
#   Solo coordina los módulos que ya construimos. Cada responsabilidad
#   vive en su módulo correspondiente:
#     hsv_utils   → detección de color
#     tracker     → seguimiento y señales
#     classifier  → reconocimiento de letra
#     word_builder→ construcción de texto
#     canvas_utils→ visualización de trayectoria
# =============================================================================

import cv2
import time
from hsv_utils import procesar_frame
from tracker import Tracker
from classifier import Classifier
from word_builder import WordBuilder
from canvas_utils import dibujar_trayectoria_en_frame
from config import (
    COLOR_TRAYECTORIA,
    COLOR_TEXTO_UI,
    COLOR_ALERTA,
    FUENTE_UI,
    ESCALA_FUENTE,
    GROSOR_FUENTE
)

# Umbral mínimo de confianza para aceptar una predicción
# Si el modelo está menos del 50% seguro, descartamos la letra
UMBRAL_CONFIANZA = 0.50


# =============================================================================
# INTERFAZ DE USUARIO
# =============================================================================

def dibujar_ui(frame, letra_detectada, confianza, word_actual,
               text_final, estado, mensaje=""):
    """
    Dibuja toda la información del sistema sobre el frame de la cámara.

    Paneles:
    - Superior: letra detectada y confianza del modelo
    - Inferior: palabra actual y texto acumulado
    - Centro: mensajes de estado (fin de palabra, baja confianza, etc.)

    Args:
        frame: numpy array BGR de la cámara
        letra_detectada: última letra reconocida o "" si ninguna aún
        confianza: float [0,1] de confianza del modelo
        word_actual: string con la palabra en construcción
        text_final: string con todo el texto acumulado
        estado: string del estado del tracker (DIBUJANDO/ESPERANDO/QUIETO)
        mensaje: string opcional para mostrar en el centro
    """
    h, w = frame.shape[:2]

    # ── Panel superior ────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 70), (30, 30, 30), -1)

    # Letra detectada (grande, a la izquierda)
    if letra_detectada:
        texto_letra = f"Letra: {letra_detectada}"
        color_conf = (0, 255, 0) if confianza >= 0.7 else (0, 200, 255)
        cv2.putText(frame, texto_letra, (10, 45),
                    FUENTE_UI, 1.2, color_conf, 2)

        # Barra de confianza
        barra_x = 200
        barra_ancho = int(confianza * 150)
        cv2.rectangle(frame, (barra_x, 25), (barra_x + 150, 45), (60, 60, 60), -1)
        cv2.rectangle(frame, (barra_x, 25), (barra_x + barra_ancho, 45), color_conf, -1)
        cv2.putText(frame, f"{confianza*100:.0f}%", (barra_x + 155, 42),
                    FUENTE_UI, 0.6, COLOR_TEXTO_UI, 1)

    # Estado del tracker (derecha)
    color_estado = {
        'DIBUJANDO': (0, 255, 0),
        'ESPERANDO': (0, 165, 255),
        'QUIETO':    (0, 255, 255)
    }.get(estado, COLOR_TEXTO_UI)

    cv2.putText(frame, estado, (w - 160, 45),
                FUENTE_UI, ESCALA_FUENTE, color_estado, GROSOR_FUENTE)

    # ── Panel inferior ────────────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h - 80), (w, h), (30, 30, 30), -1)

    # Palabra actual
    cv2.putText(frame, "Palabra:", (10, h - 50),
                FUENTE_UI, 0.55, (150, 150, 150), 1)
    cv2.putText(frame, word_actual if word_actual else "_",
                (90, h - 50), FUENTE_UI, ESCALA_FUENTE,
                (0, 255, 255), GROSOR_FUENTE)

    # Texto acumulado
    cv2.putText(frame, "Texto:", (10, h - 20),
                FUENTE_UI, 0.55, (150, 150, 150), 1)
    # Truncar si es muy largo para que quepa en pantalla
    texto_mostrar = text_final[-40:] + ("..." if len(text_final) > 40 else "")
    cv2.putText(frame, texto_mostrar if texto_mostrar else "_",
                (75, h - 20), FUENTE_UI, ESCALA_FUENTE,
                COLOR_TEXTO_UI, GROSOR_FUENTE)

    # ── Mensaje central (alertas) ─────────────────────────────────────────────
    if mensaje:
        tamaño = cv2.getTextSize(mensaje, FUENTE_UI, 0.9, 2)[0]
        x = (w - tamaño[0]) // 2
        y = h // 2
        # Sombra
        cv2.putText(frame, mensaje, (x+2, y+2), FUENTE_UI, 0.9, (0,0,0), 3)
        cv2.putText(frame, mensaje, (x, y), FUENTE_UI, 0.9, COLOR_ALERTA, 2)

    # ── Instrucciones pequeñas (esquina derecha inferior) ────────────────────
    instrucciones = [
        "Q: salir",
        "Quieto 2s: fin palabra",
        "Ausente 1.5s: fin letra"
    ]
    for i, inst in enumerate(instrucciones):
        cv2.putText(frame, inst, (w - 220, h - 80 - (i * 18)),
                    FUENTE_UI, 0.42, (100, 100, 100), 1)

    return frame


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    print("=" * 60)
    print("  SISTEMA DE RECONOCIMIENTO DE LETRAS EN TIEMPO REAL")
    print("=" * 60)

    # ── Inicializar módulos ───────────────────────────────────────────────────
    print("\nInicializando módulos...")

    try:
        clf = Classifier()
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        return

    tracker      = Tracker()
    word_builder = WordBuilder()

    # ── Inicializar cámara ────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n[OK] Sistema listo. Empieza a dibujar letras.\n")

    # ── Variables de estado de la UI ──────────────────────────────────────────
    ultima_letra     = ""
    ultima_confianza = 0.0
    mensaje_ui       = ""
    tiempo_mensaje   = 0.0
    corriendo        = True

    # ── Bucle principal ───────────────────────────────────────────────────────
    while corriendo:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)  # efecto espejo

        # Pipeline de visión
        mascara  = procesar_frame(frame)
        resultado = tracker.actualizar(mascara)

        # Dibujar trayectoria actual sobre el frame
        if resultado['trayectoria']:
            dibujar_trayectoria_en_frame(
                frame,
                resultado['trayectoria'],
                COLOR_TRAYECTORIA
            )

        # ── Procesar señales del tracker ──────────────────────────────────────
        señal = resultado['señal']

        if señal == 'FIN_LETRA':
            trayectoria = resultado['trayectoria']
            letra, confianza = clf.predecir(trayectoria)

            if confianza >= UMBRAL_CONFIANZA and letra != '?':
                word_builder.agregar_letra(letra)
                ultima_letra     = letra
                ultima_confianza = confianza
                mensaje_ui       = f"'{letra}' detectada ({confianza*100:.0f}%)"
            else:
                mensaje_ui = f"Confianza baja ({confianza*100:.0f}%), ignorada"
                print(f"  [main] Predicción ignorada: '{letra}' con {confianza:.2f}")

            tiempo_mensaje = time.time()
            tracker.resetear_trayectoria()

        elif señal == 'FIN_PALABRA':
            word_builder.fin_palabra()
            ultima_letra   = ""
            mensaje_ui     = "--- FIN DE PALABRA ---"
            tiempo_mensaje = time.time()
            tracker.resetear_trayectoria()

        # ── Limpiar mensaje después de 2 segundos ─────────────────────────────
        msg_visible = ""
        if time.time() - tiempo_mensaje < 2.0:
            msg_visible = mensaje_ui

        # ── Construir y mostrar frame final ───────────────────────────────────
        dibujar_ui(
            frame,
            ultima_letra,
            ultima_confianza,
            word_builder.obtener_word_actual(),
            word_builder.obtener_text_final(),
            resultado['estado'],
            msg_visible
        )

        cv2.imshow("Sistema de Reconocimiento", frame)
        cv2.imshow("Mascara HSV", mascara)

        # ── Tecla 'q' para salir ──────────────────────────────────────────────
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord('q'):
            print("\n[SALIR] Cerrando sistema...")
            corriendo = False

    # ── Cierre y guardado ─────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    ruta_archivo = word_builder.fin_sesion()

    print("\n" + "=" * 60)
    print("  SESIÓN FINALIZADA")
    if ruta_archivo:
        print(f"  Texto guardado en: {ruta_archivo}")
        print(f"  Texto generado:    '{word_builder.obtener_text_final().strip()}'")
    else:
        print("  No se generó texto en esta sesión.")
    print("=" * 60)


if __name__ == "__main__":
    main()
