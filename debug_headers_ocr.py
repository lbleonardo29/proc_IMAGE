import cv2
import pytesseract
import numpy as np

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def debug_headers():
    img = cv2.imread('c:/Proyecto_TPIM/proc_IMAGE/TMPI_T2_4BINGO/bingo_tablero.png')
    if img is None:
        print("Imagen no encontrada")
        return
    
    # Redimensionar para mejorar OCR
    alto, ancho = img.shape[:2]
    img_grande = cv2.resize(img, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)
    gris = cv2.cvtColor(img_grande, cv2.COLOR_BGR2GRAY)
    procesada = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    print("--- OCR OUTPUT (PSM 3) ---")
    texto3 = pytesseract.image_to_string(procesada, lang='eng', config='--psm 3')
    print(texto3[:2000])
    
    print("\n--- OCR OUTPUT (PSM 6) ---")
    texto6 = pytesseract.image_to_string(procesada, lang='eng', config='--psm 6')
    print(texto6[:2000])

if __name__ == '__main__':
    debug_headers()
