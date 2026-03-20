import cv2
import pytesseract
import re
import numpy as np
import threading
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

frame_actual = None
numeros_detectados = set()
procesando_ocr = False
MAX_NUMERO = 810
ejecutando = True
nombre_jugador = "Desconocido"
fecha_juego = "--/--/----"
numero_juego = "-"

def extraer_numeros_ocr(imagen):
    global numeros_detectados, procesando_ocr, nombre_jugador, fecha_juego, numero_juego
    procesando_ocr = True
    try:
        # 1. Mejora de la imagen para números pequeños y grids
        # Redimensionar (upscaling) ayuda a Tesseract con números pequeños
        alto, ancho = imagen.shape[:2]
        imagen_grande = cv2.resize(imagen, (ancho * 2, alto * 2), interpolation=cv2.INTER_CUBIC)
        
        gris = cv2.cvtColor(imagen_grande, cv2.COLOR_BGR2GRAY)
        
        # 2. Umbralización adaptativa ajustada para grids de bingo
        # Usamos un bloque más grande (11) y una constante (2) para no borrar trazos finos
        procesada = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        
        # 3. Configuración PSM 3: Automática (mejor para páginas con cabecera y tablas)
        config_tess = '--psm 3'
        texto_completo = pytesseract.image_to_string(procesada, lang='eng', config=config_tess)
        
        # --- EXTRACCIÓN DE METADATOS ---
        
        # Extraer Numero de Juego (Flexible: misma línea o siguiente)
        match_num = re.search(r'(?:Numero de Juego|Juego)\s*[:\-]?\s*([a-zA-Z0-9]+)', texto_completo, re.IGNORECASE)
        if match_num:
            val = match_num.group(1).lower()
            # Normalizar el '1' que a veces se lee como 'i' o 'l'
            numero_juego = "1" if val in ['i', 'l', '1'] else val
            
        # Extraer Fecha (Busca cualquier patrón DD/MM/YYYY cerca de la palabra Fecha o solo el patrón)
        match_fecha = re.search(r'Fecha[^\d]*(\d{1,2}/\d{1,2}/\d{2,4})', texto_completo, re.IGNORECASE)
        if not match_fecha:
            match_fecha = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', texto_completo)
            
        if match_fecha:
            fecha_juego = match_fecha.group(1)
            
        # Extraer Nombre (Busca texto después de Nombre de Jugador)
        match_nombre = re.search(r'Nombre de Jugador\s*[:\-]?\s*([A-Za-z\s]{3,})', texto_completo, re.IGNORECASE)
        if match_nombre:
            n = match_nombre.group(1).strip()
            nombre_jugador = ' '.join(n.split()) # Limpia espacios extra
            
        # --- EXTRACCIÓN DE NÚMEROS DEL TABLERO ---
            
        todos_los_numeros = re.findall(r'\b\d{1,3}\b', texto_completo)
        nuevos_nums = {int(n) for n in todos_los_numeros if 1 <= int(n) <= MAX_NUMERO}
        
        # En modo vivo, es mejor acumular lo que la IA lee bien cada segundo
        numeros_detectados.update(nuevos_nums)
    except Exception as e:
        print("Error OCR:", e)
    procesando_ocr = False

def hilo_ocr():
    global frame_actual, ejecutando
    while ejecutando:
        if frame_actual is not None and not procesando_ocr:
            frame_copia = frame_actual.copy()
            extraer_numeros_ocr(frame_copia)
        # Pausa leve para que la CPU respire entre frames capturados
        time.sleep(0.5)

def crear_panel_tablero(nums):
    alto, ancho = 720, 750
    tablero = np.zeros((alto, ancho, 3), dtype=np.uint8)
    tablero[:] = (30, 30, 30) # Gris super oscuro de fondo
    
    cv2.putText(tablero, "TABLERO BINGO", (250, 35), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(tablero, f"Detectados: {len(nums)} / {MAX_NUMERO}", (250, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    info_text = f"Jugador: {nombre_jugador} | Fecha: {fecha_juego} | Juego: {numero_juego}"
    cv2.putText(tablero, info_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    columnas = 30
    filas = 27
    margen_x = 20
    margen_y = 120
    ancho_celda = (ancho - 2 * margen_x) // columnas
    alto_celda = (alto - margen_y - 20) // filas
    
    for i in range(1, MAX_NUMERO + 1):
        fila = (i - 1) // columnas
        col = (i - 1) % columnas
        
        x1 = margen_x + col * ancho_celda
        y1 = margen_y + fila * alto_celda
        x2 = x1 + ancho_celda
        y2 = y1 + alto_celda
        
        if i in nums:
            color_fondo = (0, 200, 0) # Verde intenso cuando lo encuentra
            color_texto = (0, 0, 0)   # Numero negro
        else:
            color_fondo = (50, 50, 50) # Gris apagado si falta
            color_texto = (150, 150, 150) # Numero clarito
            
        cv2.rectangle(tablero, (x1, y1), (x2, y2), color_fondo, -1)
        cv2.rectangle(tablero, (x1, y1), (x2, y2), (20, 20, 20), 1) # Borde
        
        font_scale = 0.35
        grosor = 1
        (txt_w, txt_h), _ = cv2.getTextSize(str(i), cv2.FONT_HERSHEY_SIMPLEX, font_scale, grosor)
        txt_x = x1 + (ancho_celda - txt_w) // 2
        txt_y = y1 + (alto_celda + txt_h) // 2
        
        cv2.putText(tablero, str(i), (txt_x, txt_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_texto, grosor)
        
    return tablero

def modo_vivo():
    global frame_actual, ejecutando
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No se pudo iniciar la cámara.")
        return
        
    print("Iniciando modo tablero en vivo... Presiona ESC en la ventana para salir.")
    thread = threading.Thread(target=hilo_ocr, daemon=True)
    thread.start()
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_actual = frame.copy()
        
        # Panel Izquierdo: Camara / Panel Derecho: Tablero
        frame_mostrar = cv2.resize(frame, (640, 720))
        panel_mostrar = crear_panel_tablero(numeros_detectados)
        
        if procesando_ocr:
            cv2.putText(frame_mostrar, "ESCANEANDO...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
        cv2.putText(frame_mostrar, "CAMARA EN VIVO", (20, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        vista_final = np.hstack((frame_mostrar, panel_mostrar))
        
        # Check if all numbers are detected and show an alert
        if len(numeros_detectados) >= MAX_NUMERO:
            mensaje = "¡NUMEROS DETECTADOS COMPLETAMENTE!"
            # Black border for better visibility
            cv2.putText(vista_final, mensaje, (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 6)
            # Green text
            cv2.putText(vista_final, mensaje, (50, 360), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            
        cv2.imshow('Bingo Live Dashboard', vista_final)
        
        if cv2.waitKey(30) & 0xFF == 27:
            ejecutando = False
            break
            
    cap.release()
    cv2.destroyAllWindows()

def modo_estatico():
    img = cv2.imread('TMPI_T2_4BINGO/bingo_tablero.png')
    if img is not None:
        print("Escaneando imagen estática, por favor espera un momento...")
        # Limpiar detecciones previas
        numeros_detectados.clear()
        
        extraer_numeros_ocr(img)
        panel_mostrar = crear_panel_tablero(numeros_detectados)
        img_mostrar = cv2.resize(img, (640, 720))
        vista_final = np.hstack((img_mostrar, panel_mostrar))
        
        print(f"Detectados {len(numeros_detectados)} números únicos de {MAX_NUMERO}")
        cv2.imshow('Bingo Resultado (Presiona ESC en la ventana para salir)', vista_final)
        while True:
            if cv2.waitKey(100) & 0xFF == 27:
                break
        cv2.destroyAllWindows()
    else:
        print("Imagen 'TMPI_T2_4BINGO/bingo_tablero.png' no encontrada.")

if __name__ == '__main__':
    print("1. Modo Dashboard en Vivo (Cámara)")
    print("2. Modo Imagen de Prueba ('TMPI_T2_4BINGO/bingo_tablero.png')")
    op = input("Opción: ")

    if op == '1':
        modo_vivo()
    elif op == '2':
        modo_estatico()
    else:
        print("Opción inválida.")