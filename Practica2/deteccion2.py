import numpy as np
import cv2
rojo2_lower = np.array([170, 100, 100], np.uint8)
rojo2_upper = np.array([180, 255, 255], np.uint8) 

ROJO_lower = np.array([0, 100, 100], np.uint8)
ROJO_upper = np.array([10, 255, 255], np.uint8)     

azul_lower = np.array([100, 100, 100], np.uint8)
azul_upper = np.array([130, 255, 255], np.uint8)


green_lower = np.array([40, 100, 100], np.uint8)
green_upper = np.array([90, 255, 255], np.uint8)

# Rangos de amarillo ampliados y más tolerantes
yellow_lower = np.array([15, 80, 80], np.uint8)
yellow_upper = np.array([40, 255, 255], np.uint8)

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not video.isOpened():
    print("Error: No se pudo acceder a la cámara")
    exit()

kernel = np.ones((5,5), np.uint8)

while True:
    ret, frame = video.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        rojo_mask1 = cv2.inRange(hsv, ROJO_lower, ROJO_upper)
        rojo_mask2 = cv2.inRange(hsv, rojo2_lower, rojo2_upper)
        rojo_mask = cv2.bitwise_or(rojo_mask1, rojo_mask2)
        # limpieza: apertura para ruido pequeño y cierre para unir regiones
        rojo_mask = cv2.morphologyEx(rojo_mask, cv2.MORPH_OPEN, kernel)
        rojo_mask = cv2.morphologyEx(rojo_mask, cv2.MORPH_CLOSE, kernel)
        result_red = cv2.bitwise_and(frame, frame, mask=rojo_mask)
        cv2.imshow('Mascara Roja', rojo_mask)
        cv2.imshow('Resultado Rojo', result_red)


        mask_green = cv2.inRange(hsv, green_lower, green_upper)
        result_green = cv2.bitwise_and(frame, frame, mask=mask_green)
        cv2.imshow('Mascara Verde', mask_green)
        cv2.imshow('Resultado Verde', result_green)
        cv2.imshow('Video Original', frame)



        mask_blue = cv2.inRange(hsv, azul_lower, azul_upper)
        result_blue = cv2.bitwise_and(frame, frame, mask=mask_blue)
        cv2.imshow('Mascara Azul', mask_blue)
        cv2.imshow('Resultado Azul', result_blue)
        cv2.imshow('Video Original', frame)                                                                         


        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        # limpieza: apertura para ruido pequeño y cierre para unir regiones
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
        mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_CLOSE, kernel)

        result = cv2.bitwise_and(frame, frame, mask=mask_yellow)
        
        cv2.imshow('Video Original', frame)
        cv2.imshow('Mascara Amarilla', mask_yellow)
        cv2.imshow('Resultado Amarillo', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()