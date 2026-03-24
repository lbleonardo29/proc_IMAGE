# =============================================================================
# collect_data.py — Recolección del dataset de letras dibujadas con marcador
#
# CÓMO USAR:
#   1. Ejecuta: python collect_data.py
#   2. El programa te irá pidiendo que dibujes cada letra del alfabeto.
#   3. Dibuja la letra en el aire con tu marcador verde.
#   4. Cuando termines, retira el marcador ~1.5 segundos → la imagen se guarda.
#   5. Repite MUESTRAS_POR_LETRA veces por cada letra.
#   6. Presiona 'q' para salir en cualquier momento.
#
# SEÑALES:
#   - Marcador ausente ~1.5s  → fin de letra (se guarda automáticamente)
#   - Marcador quieto  ~2s    → fin de palabra (no aplica aquí, se ignora)
#   - Tecla 'q'               → salir del programa
#
# RESULTADO:
#   Carpeta dataset/raw/LETRA/ con MUESTRAS_POR_LETRA imágenes por cada letra.
# =============================================================================

import cv2
import os
import time
from config import (
    ALFABETO,
    CARPETA_SEGURA,
    MUESTRAS_POR_LETRA,
    RUTA_DATASET_RAW,
    COLOR_TRAYECTORIA,
    COLOR_TEXTO_UI,
    COLOR_ALERTA,
    FUENTE_UI,
    ESCALA_FUENTE,
    GROSOR_FUENTE
)
from hsv_utils import procesar_frame
from tracker import Tracker
from canvas_utils import guardar_imagen_letra, dibujar_trayectoria_en_frame


# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def crear_carpetas_dataset():
    """
    Crea la estructura de carpetas del dataset si no existe.

    Usa nombres de carpeta ASCII seguros (CARPETA_SEGURA) para evitar
    problemas de codificación en Windows con caracteres como 'Ñ'.
    La carpeta de la Ñ se llamará 'NTILDE' en disco, pero en pantalla
    y en el modelo siempre aparece como 'Ñ'.
    """
    for letra in ALFABETO:
        nombre_carpeta = CARPETA_SEGURA[letra]
        ruta = os.path.join(RUTA_DATASET_RAW, nombre_carpeta)
        os.makedirs(ruta, exist_ok=True)
    print(f"[OK] Carpetas del dataset listas en: {RUTA_DATASET_RAW}/")
    print(f"     Nota: la Ñ se guarda en la carpeta 'NTILDE' por compatibilidad Windows.")


def contar_muestras_existentes(letra):
    """
    Cuenta cuántas imágenes ya hay guardadas para una letra específica.

    Args:
        letra: string con la letra, por ejemplo 'A' o 'Ñ'

    Returns:
        int: número de archivos .png en la carpeta de esa letra
    """
    nombre_carpeta = CARPETA_SEGURA[letra]
    ruta = os.path.join(RUTA_DATASET_RAW, nombre_carpeta)
    if not os.path.exists(ruta):
        return 0
    archivos = [f for f in os.listdir(ruta) if f.endswith('.png')]
    return len(archivos)


def dibujar_ui(frame, letra_actual, muestra_actual, total_muestras,
               estado_tracker, mensaje_extra=""):
    """
    Dibuja toda la información de la interfaz de usuario sobre el frame.

    Centralizar el dibujo de la UI en una función separada mantiene el
    bucle principal limpio y fácil de leer. Toda la "cosmética" está aquí.

    Args:
        frame: el frame de la cámara donde se dibujará la UI
        letra_actual: string con la letra que se está recolectando
        muestra_actual: int, número de muestra actual (ej: 3)
        total_muestras: int, total de muestras a recolectar (ej: 10)
        estado_tracker: string con el estado de la máquina del tracker
        mensaje_extra: string opcional para mostrar alertas o instrucciones
    """
    h, w = frame.shape[:2]

    # ── Panel superior: letra actual y progreso ───────────────────────────────
    cv2.rectangle(frame, (0, 0), (w, 60), (40, 40, 40), -1)  # fondo oscuro

    texto_letra = f"Dibuja la letra: {letra_actual}"
    texto_progreso = f"Muestra {muestra_actual}/{total_muestras}"

    cv2.putText(frame, texto_letra, (10, 30),
                FUENTE_UI, ESCALA_FUENTE + 0.3, COLOR_TEXTO_UI, GROSOR_FUENTE)
    cv2.putText(frame, texto_progreso, (w - 180, 30),
                FUENTE_UI, ESCALA_FUENTE, COLOR_TEXTO_UI, GROSOR_FUENTE)

    # ── Panel inferior: estado del tracker ───────────────────────────────────
    cv2.rectangle(frame, (0, h - 50), (w, h), (40, 40, 40), -1)

    color_estado = COLOR_ALERTA if estado_tracker == "ESPERANDO" else (0, 200, 0)
    cv2.putText(frame, f"Estado: {estado_tracker}", (10, h - 20),
                FUENTE_UI, ESCALA_FUENTE, color_estado, GROSOR_FUENTE)

    # ── Mensaje extra centrado en pantalla (solo cuando hay algo que decir) ──
    if mensaje_extra:
        tamaño_texto = cv2.getTextSize(
            mensaje_extra, FUENTE_UI, ESCALA_FUENTE + 0.2, GROSOR_FUENTE
        )[0]
        x_centro = (w - tamaño_texto[0]) // 2
        # Sombra para mejor legibilidad
        cv2.putText(frame, mensaje_extra, (x_centro + 1, h // 2 + 1),
                    FUENTE_UI, ESCALA_FUENTE + 0.2, (0, 0, 0), GROSOR_FUENTE + 1)
        cv2.putText(frame, mensaje_extra, (x_centro, h // 2),
                    FUENTE_UI, ESCALA_FUENTE + 0.2, COLOR_ALERTA, GROSOR_FUENTE)

    return frame


# =============================================================================
# PROGRAMA PRINCIPAL
# =============================================================================

def main():
    """
    Bucle principal de recolección de datos.

    La lógica general es:
    1. Inicializar cámara, tracker y carpetas
    2. Iterar por cada letra del alfabeto
    3. Para cada letra, recolectar MUESTRAS_POR_LETRA imágenes
    4. En cada frame: procesar → trackear → detectar señales → guardar si corresponde
    5. Al terminar todas las letras, mostrar resumen y cerrar
    """
    print("=" * 60)
    print("  RECOLECTOR DE DATASET — Visión Artificial")
    print("=" * 60)
    print(f"  Letras a recolectar: {len(ALFABETO)}")
    print(f"  Muestras por letra:  {MUESTRAS_POR_LETRA}")
    print(f"  Total de imágenes:   {len(ALFABETO) * MUESTRAS_POR_LETRA}")
    print(f"  Guardado en:         {RUTA_DATASET_RAW}/")
    print("=" * 60)
    print("\nInstrucciones:")
    print("  - Dibuja la letra mostrada con tu marcador VERDE")
    print("  - Retira el marcador 1.5 segundos para confirmar la letra")
    print("  - Presiona 'q' para salir\n")

    # ── Crear estructura de carpetas ─────────────────────────────────────────
    crear_carpetas_dataset()

    # ── Inicializar cámara ───────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara. Verifica que esté conectada.")
        return

    # Configurar resolución de la cámara
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    tracker = Tracker()
    mensaje_ui = ""
    tiempo_mensaje = 0  # timestamp de cuándo se activó el último mensaje UI

    print("\nCámara lista. La ventana se abrirá ahora.\n")

    # ── Iterar por cada letra del alfabeto ───────────────────────────────────
    for letra in ALFABETO:
        muestras_recolectadas = contar_muestras_existentes(letra)

        if muestras_recolectadas >= MUESTRAS_POR_LETRA:
            print(f"  [SALTAR] {letra}: ya tiene {muestras_recolectadas} muestras.")
            continue

        print(f"\n  Recolectando letra: {letra}")
        print(f"  Ya tienes: {muestras_recolectadas}/{MUESTRAS_POR_LETRA} muestras")

        tracker.resetear_trayectoria()

        # ── Bucle de recolección para esta letra ─────────────────────────────
        while muestras_recolectadas < MUESTRAS_POR_LETRA:
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] No se pudo leer el frame de la cámara.")
                break

            # Voltear horizontalmente para efecto "espejo"
            # (más intuitivo para el usuario: sus movimientos coinciden con
            # lo que ve en pantalla, como un espejo real)
            frame = cv2.flip(frame, 1)

            # Pipeline de visión: frame → máscara → resultado del tracker
            mascara = procesar_frame(frame)
            resultado = tracker.actualizar(mascara)

            # Dibujar la trayectoria acumulada sobre el frame
            if resultado['trayectoria']:
                dibujar_trayectoria_en_frame(
                    frame,
                    resultado['trayectoria'],
                    COLOR_TRAYECTORIA
                )

            # ── Detectar señal de FIN DE LETRA ───────────────────────────────
            if resultado['señal'] == 'FIN_LETRA':
                trayectoria = resultado['trayectoria']

                # Generar nombre de archivo con número consecutivo
                nombre_carpeta = CARPETA_SEGURA[letra]
                nombre_archivo = f"muestra_{muestras_recolectadas + 1:03d}.png"
                ruta_completa = os.path.join(
                    RUTA_DATASET_RAW, nombre_carpeta, nombre_archivo
                )

                # Guardar imagen
                guardado = guardar_imagen_letra(trayectoria, ruta_completa)

                if guardado:
                    muestras_recolectadas += 1
                    print(f"    [GUARDADO] {nombre_archivo} "
                          f"({muestras_recolectadas}/{MUESTRAS_POR_LETRA})")
                    mensaje_ui = f"¡Guardada! {muestras_recolectadas}/{MUESTRAS_POR_LETRA}"
                else:
                    mensaje_ui = "Trazo muy corto, intenta de nuevo"

                tiempo_mensaje = time.time()
                tracker.resetear_trayectoria()

            # ── Mostrar mensaje UI por 2 segundos ────────────────────────────
            msg_visible = ""
            if time.time() - tiempo_mensaje < 2.0:
                msg_visible = mensaje_ui

            # ── Construir y mostrar el frame final ───────────────────────────
            dibujar_ui(
                frame,
                letra,
                muestras_recolectadas,
                MUESTRAS_POR_LETRA,
                resultado['estado'],
                msg_visible
            )

            # Mostrar también la máscara en una ventana secundaria
            # (útil para verificar que el marcador verde se detecta bien)
            cv2.imshow("Vista principal", frame)
            cv2.imshow("Mascara HSV", mascara)

            # Salir con 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[SALIR] El usuario presionó 'q'. Cerrando...")
                cap.release()
                cv2.destroyAllWindows()
                return

        print(f"  [COMPLETO] Letra '{letra}': {muestras_recolectadas} muestras guardadas.")

    # ── Finalización ─────────────────────────────────────────────────────────
    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("  ¡RECOLECCIÓN COMPLETA!")
    print(f"  Dataset guardado en: {RUTA_DATASET_RAW}/")
    print("  Siguiente paso: ejecuta python augment_data.py")
    print("=" * 60)


if __name__ == "__main__":
    main()