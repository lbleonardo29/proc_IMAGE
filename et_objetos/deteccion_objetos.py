import numpy as np
import cv2

def figcolor(figura_hsv):
    rojo2_lower = np.array([170, 100, 100], np.uint8)
    rojo2_upper = np.array([180, 255, 255], np.uint8) 

    ROJO_lower = np.array([0, 100, 100], np.uint8)
    ROJO_upper = np.array([10, 255, 255], np.uint8)     

    azul_lower = np.array([100, 100, 100], np.uint8)
    azul_upper = np.array([130, 255, 255], np.uint8)

    green_lower = np.array([40, 100, 100], np.uint8)
    green_upper = np.array([90, 255, 255], np.uint8)

    yellow_lower = np.array([20, 100, 100], np.uint8)
    yellow_upper = np.array([30, 255, 255], np.uint8)

    rojo_mask1 = cv2.inRange(figura_hsv, ROJO_lower, ROJO_upper)
    rojo_mask2 = cv2.inRange(figura_hsv, rojo2_lower, rojo2_upper)
    rojo_mask = cv2.bitwise_or(rojo_mask1, rojo_mask2)

    mask_green = cv2.inRange(figura_hsv, green_lower, green_upper)
    mask_blue = cv2.inRange(figura_hsv, azul_lower, azul_upper)
    mask_yellow = cv2.inRange(figura_hsv, yellow_lower, yellow_upper)

    cntr_red = cv2.findContours(rojo_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    cntr_green = cv2.findContours(mask_green, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    cntr_blue = cv2.findContours(mask_blue, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]
    cntr_yellow = cv2.findContours(mask_yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[0]

    color_detected = 'X'
    if len(cntr_red) > 0:
        color_detected = 'Rojo'
    elif len(cntr_green) > 0:
        color_detected = 'Verde'
    elif len(cntr_blue) > 0:
        color_detected = 'Azul'
    elif len(cntr_yellow) > 0:
        color_detected = 'Amarillo'
    
    return color_detected

def figura(width, height, contours):
    nombre_figura = 'X'
    epsilon = 0.01 * cv2.arcLength(contours, True)
    approx = cv2.approxPolyDP(contours, epsilon, True)
    
    if len(approx) == 3:
        nombre_figura = 'Triangulo'
    elif len(approx) == 4:
        aspect_ratio = float(width) / height
        if 0.95 <= aspect_ratio <= 1.05:
            nombre_figura = 'Cuadrado'
        else:
            nombre_figura = 'Rectangulo'    
    elif len(approx) == 5:
        nombre_figura = 'Pentagono'
    elif len(approx) == 6:
        nombre_figura = 'Hexagono'
    elif len(approx) > 10:
        nombre_figura = 'Circulo'
    
    return nombre_figura


# Código principal
img = cv2.imread('Practica2/figura.png')
gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gris, 50, 150)
kernel = np.ones((3, 3), np.uint8)
canny = cv2.dilate(canny, kernel, iterations=1)
contorno = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

for c in contorno[0]:
    x, y, w, h = cv2.boundingRect(c)
    img_aux = np.zeros(img_hsv.shape, dtype=np.uint8)
    cv2.drawContours(img_aux, [c], -1, (255, 255, 255), -1)
    mask_hsv = cv2.bitwise_and(img_hsv, img_aux)
    
    fill_all = figura(w, h, c) + ' ' + figcolor(mask_hsv)
    cv2.putText(img, fill_all, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()