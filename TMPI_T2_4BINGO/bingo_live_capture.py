import cv2
import pytesseract
import re

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def procesar_bingo(imagen):
    # Preprocesamiento
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    procesada = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
    
    # Extracción de datos
    texto_completo = pytesseract.image_to_string(procesada, lang='eng', config='--psm 3')
    
    todos_los_numeros = re.findall(r'\b\d{1,3}\b', texto_completo)
    
    lista_final = [n for n in todos_los_numeros if 1 <= int(n) <= 810]
    
    print("\n" + "="*40)
    print("DATOS DE CABECERA DETECTADOS:")
    lineas = texto_completo.split('\n')
    for linea in lineas[:5]: 
        if len(linea.strip()) > 5: print(f"> {linea.strip()}")
    
    print(f"\nSe detectaron {len(set(lista_final))} números únicos en el tablero.")
    print("="*40)

    # SELECCIÓN DE N NÚMEROS
    entrada = input("\nSelecciona tus números (separados por coma, ej: 1, 45, 700): ")
    seleccionados = [s.strip() for s in entrada.split(',')]
    
    print("\n--- RESULTADO DE SELECCIÓN ---")
    for num in seleccionados:
        if num in lista_final:
            print(f"Número {num}: ENCONTRADO")
        else:
            print(f"Número {num}: NO detectado")

# --- MENÚ DE INICIO ---
print("1. Capturar con Cámara\n2. Cargar 'TMPI_T2_4BINGO/bingo_tablero.jpg'")
op = input("Opción: ")
if op == '1':
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        cv2.imshow('Captura Bingo', frame)
        if cv2.waitKey(1) & 0xFF == 32:
            procesar_bingo(frame)
            break
    cap.release()
else:
    img = cv2.imread('TMPI_T2_4BINGO/bingo_tablero.jpg')
    if img is not None: procesar_bingo(img)
cv2.destroyAllWindows()