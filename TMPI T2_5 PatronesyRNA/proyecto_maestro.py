import cv2
import numpy as np
import os

# Obtenemos la ruta absoluta de la carpeta donde está este script
base_dir = os.path.dirname(os.path.abspath(__file__))

print("\n--- Iniciando Reconocimiento de Patrones Clásico ---")

# 1. Cargar la escena (Nota: Ajusté el nombre a lo que sale en tu captura)
ruta_rp = os.path.join(base_dir, 'autos_rp.png')
escena_rp = cv2.imread(ruta_rp) 

if escena_rp is None:
    print(f"Error: No se encontró 'autos_rp.png' en {ruta_rp}. Revisa el nombre en la carpeta.")
else:
    gris_escena = cv2.cvtColor(escena_rp, cv2.COLOR_BGR2GRAY)

    # --- AQUÍ ESTÁ LA CORRECCIÓN ---
    # Definimos el tamaño del patrón (Ancho y Alto)
    w_p, h_p = 220, 130 
    
    # Extraemos el "molde" o patrón de la imagen original usando Slicing
    patron_gris = gris_escena[70:70+h_p, 210:210+w_p]

    # Ejecutamos el Template Matching
    resultado = cv2.matchTemplate(gris_escena, patron_gris, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(resultado)

    resultado_rp = escena_rp.copy()
    cv2.rectangle(resultado_rp, max_loc, (max_loc[0] + w_p, max_loc[1] + h_p), (0, 255, 0), 4)
    
    label = f"Match de Patron: {max_val:.2f}%"
    cv2.putText(resultado_rp, label, (max_loc[0], max_loc[1]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    cv2.imshow('1. Reconocimiento de Patrones Clasico', resultado_rp)

# Inicio de Red Neuronal Artificial (RNA)
print("\n--- Iniciando Red Neuronal Artificial ---")

ruta_rna = os.path.join(base_dir, 'autos_rna.jpg')
escena_rna = cv2.imread(ruta_rna)

if escena_rna is None:
    print(f"Error: No se encontro 'autos_rna.jpg' en {ruta_rna}. Deteniendo Parte B.")
else:
    clases = ["background", "aeroplane", "bicycle", "bird", "boat",
              "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
              "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
              "sofa", "train", "tvmonitor"]

    prototxt = os.path.join(base_dir, "MobileNetSSD_deploy.prototxt")
    model = os.path.join(base_dir, "MobileNetSSD_deploy.caffemodel")
    
    try:
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
    except Exception as e:
        print(f"Error cargando archivos de RNA: {e}. Revisa sus nombres.")
    else:
        (h_img, w_img) = escena_rna.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(escena_rna, (300, 300)), 0.007843, (300, 300), 127.5)
        
        net.setInput(blob)
        detecciones = net.forward()

        for i in range(detecciones.shape[2]):
            confianza = detecciones[0, 0, i, 2]
            if confianza > 0.6: 
                idx = int(detecciones[0, 0, i, 1])
                
                if clases[idx] in ["car", "bus"]:
                    caja = detecciones[0, 0, i, 3:7] * np.array([w_img, h_img, w_img, h_img])
                    (startX, startY, endX, endY) = caja.astype("int")
                    
                    label_rna = f"{clases[idx]}: {confianza:.2f}%"
                    cv2.rectangle(escena_rna, (startX, startY), (endX, endY), (0, 0, 255), 3)
                    cv2.putText(escena_rna, label_rna, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        print("Red Neuronal finalizo la deteccion sobre la imagen real.")
        cv2.imshow('2. Red Neuronal Artificial (Detección Robustar)', cv2.resize(escena_rna, (900, 600)))

print("\n--- Ejecución completada. Presiona cualquier tecla para cerrar. ---")
cv2.waitKey(0)
cv2.destroyAllWindows()