import numpy as np
import cv2

yellow_lower = np.array([20, 100, 100], np.uint8)
yellow_upper = np.array([30, 255, 255], np.uint8)

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not video.isOpened():
    print("Error: No se pudo acceder a la cámara")
    exit()

while True:
    ret, frame = video.read()
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask_yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)
        result = cv2.bitwise_and(frame, frame, mask=mask_yellow)
        
        cv2.imshow('Video Original', frame)
        cv2.imshow('Mascara Amarilla', mask_yellow)
        cv2.imshow('Resultado', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

video.release()
cv2.destroyAllWindows()