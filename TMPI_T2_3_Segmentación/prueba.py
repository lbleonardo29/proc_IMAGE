import cv2
import numpy as np

# Cargar imagen en color y en escala de grises
img = cv2.imread('paisaje.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rows, cols = img.shape[:2]
M = cv2.getRotationMatrix2D((cols/2, rows/2), 45, 1)
img_rotada = cv2.warpAffine(img, M, (cols, rows))

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
rojo_bajo = np.array([0, 100, 100])
rojo_alto = np.array([10, 255, 255])
mascara = cv2.inRange(hsv, rojo_bajo, rojo_alto)
segmentado = cv2.bitwise_and(img, img, mask=mascara)

ret, binarizada_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
adaptativa = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY, 11, 2)

cv2.imshow('1. Original', img)
cv2.imshow('2. Transformacion (Rotacion 45)', img_rotada)
cv2.imshow('3. Segmentacion (Filtro Rojo)', segmentado)
cv2.imshow('4. Binarizacion (Otsu)', binarizada_otsu)
cv2.imshow('5. Umbral Adaptativo', adaptativa)

print("Procesamiento completado. Presiona cualquier tecla en las ventanas para cerrar.")

cv2.waitKey(0)
cv2.destroyAllWindows()
