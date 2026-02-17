import cv2
import numpy as np

image = cv2.imread('pers.png')
if image is None:
    print("Error: No se pudo cargar la imagen")
    exit()


else:
    # filtro suavizado 
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    cv2.imshow('Imagen Original', image)
    cv2.imshow('Imagen Suavizada', blurred)

    # filtro negativo
    row, col, channel = image.shape
    negative = np.zeros((row, col, 3), dtype=np.uint8)
    for i in range(row):
        for j in range(col):
            negative[i, j] = 255 - image[i, j]
    cv2.imshow('Imagen Negativa', negative)
    cv2.waitKey(0)

#filtro pencilSketch
gray, color = cv2.pencilSketch(image, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
cv2.imshow('Pencil Sketch - Grayscale', gray)
cv2.imshow('Pencil Sketch - Color', color)

#filtro sepia 
copia= image.copy()
copia = cv2.transform(copia, np.matrix([[0.393, 0.769, 0.189],
                                           [0.349, 0.686, 0.168],
                                           [0.272, 0.534, 0.131]]))
copia[np.where(copia > 255)] = 255
copia = copia.astype(np.uint8)
cv2.imshow('Imagen Sepia', copia)
cv2.waitKey(0)

#filtro cartoon

borde= cv2.Canny(image, 100, 200)
gris = cv2.cvtColor(borde, cv2.COLOR_GRAY2BGR)
gris = cv2.medianBlur(gris, 5)
Borde2 = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

dts= cv2.edgePreservingFilter(image, flags=2, sigma_s=50, sigma_r=0.25)
cartton1= cv2.bitwise_and(dts, dts, mask=borde)
cartton2= cv2.bitwise_and(dts, dts, mask=Borde2)
cv2.imshow('Cartoon - Canny', cartton1)
cv2.imshow('Cartoon - Adaptive Threshold', cartton2)
cv2.waitKey(0)

#filtro sobel
Grandient_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
Grandient_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
gradient_magnitude = cv2.magnitude(Grandient_x, Grandient_y)
cv2.imshow('Sobel - Magnitud del Gradiente', gradient_magnitude)
cv2.waitKey(0)