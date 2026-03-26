import cv2
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))

def dibujar_label(img, texto, x, y, color_texto=(255, 255, 255), color_fondo=(0, 0, 0)):
    """Dibuja texto con fondo oscuro semitransparente para mejor legibilidad."""
    (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    overlay = img.copy()
    cv2.rectangle(overlay, (x - 4, y - th - 8), (x + tw + 4, y + 4), color_fondo, -1)
    cv2.addWeighted(overlay, 0.6, img, 0.4, 0, img)
    cv2.putText(img, texto, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_texto, 2)

def agregar_banner(img, texto, color_fondo=(20, 20, 20), color_texto=(255, 255, 255)):
    """Agrega una franja de título en la parte superior de la imagen."""
    h, w = img.shape[:2]
    banner = np.zeros((40, w, 3), dtype=np.uint8)
    banner[:] = color_fondo
    (tw, _), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    x_center = max(0, (w - tw) // 2)
    cv2.putText(banner, texto, (x_center, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_texto, 2)
    return np.vstack([banner, img])

# ─────────────────────────────────────────────────
#  PARTE A: Reconocimiento de Patrones (Template Matching)
# ─────────────────────────────────────────────────
print("\n--- Iniciando Reconocimiento de Patrones Clasico ---")

img_rp = None
ruta_rp = os.path.join(base_dir, 'autos_rp.png')
escena_rp = cv2.imread(ruta_rp)

if escena_rp is None:
    print(f"Error: No se encontró 'autos_rp.png' en {ruta_rp}.")
else:
    gris_escena = cv2.cvtColor(escena_rp, cv2.COLOR_BGR2GRAY)

    w_p, h_p = 220, 130
    patron_gris = gris_escena[70:70+h_p, 210:210+w_p]

    # Mostrar el patrón que se va a buscar
    patron_vis = cv2.cvtColor(patron_gris, cv2.COLOR_GRAY2BGR)
    patron_vis = cv2.resize(patron_vis, (440, 260))
    patron_vis = agregar_banner(patron_vis, "Patron Buscado (Template)")
    cv2.imshow('Patron Buscado', patron_vis)

    resultado = cv2.matchTemplate(gris_escena, patron_gris, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(resultado)

    resultado_rp = escena_rp.copy()
    cv2.rectangle(resultado_rp, max_loc, (max_loc[0] + w_p, max_loc[1] + h_p), (0, 220, 0), 4)

    label_rp = f"Match: {max_val * 100:.1f}%"
    dibujar_label(resultado_rp, label_rp, max_loc[0], max_loc[1] - 12, (0, 255, 0), (0, 60, 0))

    resultado_rp = cv2.resize(resultado_rp, (900, 600))
    resultado_rp = agregar_banner(resultado_rp, "Parte A: Reconocimiento de Patrones Clasico (Template Matching)", (0, 80, 0))
    cv2.imshow('1. Reconocimiento de Patrones Clasico', resultado_rp)
    img_rp = resultado_rp
    print(f"  -> Match de patron encontrado: {max_val * 100:.1f}%")

# ─────────────────────────────────────────────────
#  PARTE B: Red Neuronal Artificial (MobileNetSSD)
# ─────────────────────────────────────────────────
print("\n--- Iniciando Red Neuronal Artificial ---")

img_rna = None
ruta_rna = os.path.join(base_dir, 'autos_rna.jpg')
escena_rna = cv2.imread(ruta_rna)

if escena_rna is None:
    print(f"Error: No se encontro 'autos_rna.jpg' en {ruta_rna}.")
else:
    clases = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
              "sofa", "train", "tvmonitor"]

    prototxt = os.path.join(base_dir, "MobileNetSSD_deploy.prototxt")
    model    = os.path.join(base_dir, "MobileNetSSD_deploy.caffemodel")

    try:
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
    except Exception as e:
        print(f"Error cargando archivos de RNA: {e}.")
    else:
        (h_img, w_img) = escena_rna.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(escena_rna, (300, 300)),
                                     0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detecciones = net.forward()

        n_detectados = 0
        for i in range(detecciones.shape[2]):
            confianza = detecciones[0, 0, i, 2]
            if confianza > 0.6:
                idx = int(detecciones[0, 0, i, 1])
                if clases[idx] in ["car", "bus"]:
                    caja = detecciones[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
                    (startX, startY, endX, endY) = caja.astype("int")
                    label_rna = f"{clases[idx]}: {confianza * 100:.1f}%"
                    cv2.rectangle(escena_rna, (startX, startY), (endX, endY), (0, 0, 220), 3)
                    dibujar_label(escena_rna, label_rna, startX, max(startY - 10, 20),
                                  (255, 255, 255), (0, 0, 160))
                    n_detectados += 1

        escena_rna = cv2.resize(escena_rna, (900, 600))
        escena_rna = agregar_banner(escena_rna,
                                    f"Parte B: Red Neuronal Artificial — MobileNetSSD  ({n_detectados} vehiculo(s))",
                                    (120, 0, 0))
        cv2.imshow('2. Red Neuronal Artificial (Deteccion Robusta)', escena_rna)
        img_rna = escena_rna
        print(f"  -> Detecciones con confianza >60%: {n_detectados} vehiculo(s)")

# ─────────────────────────────────────────────────
#  PANEL COMPARATIVO
# ─────────────────────────────────────────────────
if img_rp is not None and img_rna is not None:
    h_target = 500
    def redim(img):
        h, w = img.shape[:2]
        escala = h_target / h
        return cv2.resize(img, (int(w * escala), h_target))

    panel_rp  = redim(img_rp)
    panel_rna = redim(img_rna)

    # Línea divisoria
    div = np.zeros((h_target, 6, 3), dtype=np.uint8)
    div[:] = (200, 200, 200)

    comparativa = np.hstack([panel_rp, div, panel_rna])
    comparativa = agregar_banner(comparativa,
                                 "Comparativa: Patrones Clasicos  vs  Red Neuronal Artificial",
                                 (40, 40, 40))
    cv2.imshow('Comparativa: RP vs RNA', comparativa)

print("\n--- Ejecucion completada. Presiona cualquier tecla para cerrar. ---")
cv2.waitKey(0)
cv2.destroyAllWindows()
