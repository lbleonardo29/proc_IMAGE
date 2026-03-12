import cv2
import sys

nombre_archivo = 'paisaje.png'
imagen = cv2.imread(nombre_archivo)
if imagen is None:
    print(f"Error: No se pudo encontrar {nombre_archivo}")
    sys.exit()

cv2.imshow('Imagen Original - Entrada', imagen)
alto, ancho, canales = imagen.shape
print(f"Dimensiones: {ancho}x{alto} con {canales} canales de color.")

cv2.waitKey(0)
cv2.destroyAllWindows()