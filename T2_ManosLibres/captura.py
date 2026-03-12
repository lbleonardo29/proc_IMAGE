import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# Definimos el rango del color que queremos detectar (en este caso, un verde)
# Estos valores son: [Hue, Saturation, Value]
piel_bajo = np.array([0, 20, 70])
piel_alto = np.array([20, 255, 255])
while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Convertir de BGR a HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv,   piel_bajo, piel_alto)
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in contornos:
        area = cv2.contourArea(c)
        if area > 1000: # Filtramos ruidos pequeños
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            
            M = cv2.moments(c)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.putText(frame, "Mano/Objeto Detectado", (cx-50, cy-20), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    # Mostrar las dos ventanas para comparar
    cv2.imshow('Mascara (Molde de Color)', mask)
    cv2.imshow('Deteccion de Contornos - Leonardo', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()