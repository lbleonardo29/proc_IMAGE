import numpy as np
import cv2

img = cv2.imread('paisaje.png')

cv2.imshow('paisaje', img)
cv2.waitKey(0)

gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Imagen en gris', gris)
cv2.waitKey(0)

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow('Imagen en HSV', hsv)
cv2.waitKey(0)

yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
cv2.imshow('Imagen en YUV', yuv)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Dividir la imagen en canales de color RGB. 
b,g,r = cv2.split(img)
cv2.imshow('Canal Rojo', r)
cv2.imshow('Canal Verde', g)
cv2.imshow('Canal Azul', b)
cv2.waitKey(0)

#Fusionar los canales de color en una imagen. 
srv = cv2.merge([b,g,r])
cv2.imshow('Imagen fusionada', srv)
cv2.waitKey(0)
cv2.destroyAllWindows()
