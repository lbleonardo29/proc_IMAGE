import cv2
import pytesseract
import numpy as np

# Configuración de ruta (Cámbiala si tu instalación de Tesseract es distinta)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def capturar_y_leer():
    cap = cv2.VideoCapture(0)
    print("--- MODO CAPTURA ACTIVO ---")
    print("1. Enfoca la hoja de Bingo")
    print("2. Presiona ESPACIO para capturar y leer OCR")
    print("3. Presiona ESC para salir")

    while True:
        ret, frame = cap.read()
        if not ret: break

        cv2.imshow('Cámara Bingo - Anti Gravity', frame)
        
        key = cv2.waitKey(1)
        if key == 32: # Tecla Espacio
            # Guardamos la captura
            cv2.imwrite('captura_bingo.png', frame)
            
            # PROCESAMIENTO
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Umbral adaptativo para manejar fondos de colores (instrucción paso 3)
            processed = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 11, 2)
            
            # OCR con configuración para bloques de texto (PSM 6)
            texto = pytesseract.image_to_string(processed, config='--psm 6', lang='spa')
            
            print("\n--- TEXTO DETECTADO ---")
            print(texto)
            
            cv2.imshow('Resultado Preprocesamiento', processed)
            print("\nCaptura realizada. Presiona cualquier tecla para continuar o ESC para salir.")
            cv2.waitKey(0)
            cv2.destroyWindow('Resultado Preprocesamiento')
            
        elif key == 27: # Tecla ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capturar_y_leer()