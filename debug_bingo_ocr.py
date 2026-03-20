import cv2
import pytesseract
import re
import numpy as np

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def test_ocr(image_path):
    img = cv2.imread(image_path)
    if img is None:
        print(f"No se pudo cargar la imagen: {image_path}")
        return

    # Probar diferentes preprocesamientos
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Preprocesamiento 1: Original
    proc1 = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
    
    # Preprocesamiento 2: Otsu (suele ser bueno para texto claro)
    _, proc2 = cv2.threshold(gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Preprocesamiento 3: Redimensionar para mejorar OCR de números pequeños
    img_resized = cv2.resize(gris, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    proc3 = cv2.adaptiveThreshold(img_resized, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    configs = ['--psm 3', '--psm 6', '--psm 11']
    procs = [('Original', proc1), ('Otsu', proc2), ('Resized', proc3)]

    for name, proc in procs:
        print(f"\n--- Probando {name} ---")
        cv2.imwrite(f'debug_{name}.png', proc)
        for config in configs:
            texto = pytesseract.image_to_string(proc, lang='eng', config=config)
            numeros = re.findall(r'\b\d{1,3}\b', texto)
            validos = [n for n in numeros if 1 <= int(n) <= 810]
            print(f"Config {config}: Detectados {len(validos)} números únicos.")
            if len(validos) > 0:
                print(f"Muestra: {validos[:10]}")

if __name__ == '__main__':
    test_ocr('c:/Proyecto_TPIM/proc_IMAGE/TMPI_T2_4BINGO/bingo_tablero.png')
