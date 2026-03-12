import cv2
import numpy as np
import matplotlib.pyplot as plt

def nada(x):
    pass

# Crear ventana con barras de ajuste (Trackbars)
cv2.namedWindow("Ajustes")
cv2.createTrackbar("H Min", "Ajustes", 0, 179, nada)
cv2.createTrackbar("S Min", "Ajustes", 0, 255, nada)
cv2.createTrackbar("V Min", "Ajustes", 0, 255, nada)
cv2.createTrackbar("H Max", "Ajustes", 179, 179, nada)
cv2.createTrackbar("S Max", "Ajustes", 255, 255, nada)
cv2.createTrackbar("V Max", "Ajustes", 255, 255, nada)
cv2.createTrackbar("Bordes", "Ajustes", 100, 500, nada) # Umbral para Canny

cap = cv2.VideoCapture(0)

print("Iniciando modo de prueba. Ajusta las barras para detectar tu mano.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # 1. Preparar Espacios de Color
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Leer valores de las barras deslizantes
    h_min = cv2.getTrackbarPos("H Min", "Ajustes")
    s_min = cv2.getTrackbarPos("S Min", "Ajustes")
    v_min = cv2.getTrackbarPos("V Min", "Ajustes")
    h_max = cv2.getTrackbarPos("H Max", "Ajustes")
    s_max = cv2.getTrackbarPos("S Max", "Ajustes")
    v_max = cv2.getTrackbarPos("V Max", "Ajustes")
    umbral_bordes = cv2.getTrackbarPos("Bordes", "Ajustes")

    # 3. Máscara de Color (Paso 3 y 4)
    bajo = np.array([h_min, s_min, v_min])
    alto = np.array([h_max, s_max, v_max])
    mask = cv2.inRange(hsv, bajo, alto)

    # 4. Detección de Bordes - Algoritmo Canny (Paso 5)
    # El algoritmo de Canny detecta cambios bruscos de intensidad
    bordes = cv2.Canny(gris, umbral_bordes / 2, umbral_bordes)

    # 5. Dibujar contornos sobre el frame original
    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contornos:
        if cv2.contourArea(c) > 2000:
            cv2.drawContours(frame, [c], -1, (255, 0, 255), 2)

    # Mostrar resultados
    cv2.imshow("1. Video Original con Contornos", frame)
    cv2.imshow("2. Mascara de Color (Molde)", mask)
    cv2.imshow("3. Deteccion de Bordes (Canny)", bordes)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # PASO 6: Si presionas 'h', generamos el histograma del momento
    elif key == ord('h'):
        plt.figure(figsize=(10,4))
        colores = ('b', 'g', 'r')
        for i, col in enumerate(colores):
            hist = cv2.calcHist([frame], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        plt.title("Histograma de Distribución de Color")
        plt.show()

cap.release()
cv2.destroyAllWindows()