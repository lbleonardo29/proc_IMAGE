import cv2
import numpy as np

# --- 1. FUNCIÓN PARA DIBUJAR EL HISTOGRAMA EN TIEMPO REAL ---
def mostrar_histograma(imagen, nombre_ventana="Histograma"):
    # Si la imagen tiene color, la pasamos a gris para el histograma
    if len(imagen.shape) == 3:
        gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    else:
        gris = imagen
        
    # Calculamos el histograma
    hist = cv2.calcHist([gris], [0], None, [256], [0, 256])
    
    # Normalizamos los valores para que quepan en una ventana de 256x256 píxeles
    cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    
    # Creamos un lienzo negro para dibujar la gráfica
    lienzo_hist = np.zeros((256, 256), dtype=np.uint8)
    
    # Dibujamos las líneas del histograma
    for x in range(256):
        cv2.line(lienzo_hist, (x, 256), (x, 256 - int(hist[x])), 255, 1)
        
    cv2.imshow(nombre_ventana, lienzo_hist)

# --- 2. CONFIGURACIÓN DE LA CÁMARA Y LOS FILTROS ---
cap = cv2.VideoCapture(0)
modo_filtro = '1' # Empezamos con el filtro 1 por defecto

print("--- CONTROLES ---")
print("Presiona 1: Filtro Bilateral (Piel Suave)")
print("Presiona 2: Filtro de Mapa de Color (Mapa de Calor JET)")
print("Presiona 3: Filtro Morfológico (Dilatación)")
print("Presiona 4: Filtro de Scharr (Bordes de alta precisión)")
print("Presiona 5: Filtro de Relieve (Emboss)")
print("Presiona Q: Salir del programa")

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Reflejamos el frame para modo "espejo"
    frame = cv2.flip(frame, 1)
    resultado = frame.copy()

    # --- 3. APLICACIÓN DE LOS 5 FILTROS INVESTIGADOS ---
    if modo_filtro == '1':
        # Filtro Bilateral: Reduce ruido preservando bordes
        resultado = cv2.bilateralFilter(frame, 15, 80, 80)
        cv2.putText(resultado, "1. Filtro Bilateral", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif modo_filtro == '2':
        # Filtro Colormap: Asigna colores basados en la intensidad (Mapa de calor)
        resultado = cv2.applyColorMap(frame, cv2.COLORMAP_JET)
        cv2.putText(resultado, "2. Mapa de Calor (JET)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    elif modo_filtro == '3':
        # Filtro Morfológico (Dilatación): Engrosa áreas brillantes
        kernel = np.ones((5,5), np.uint8)
        resultado = cv2.dilate(frame, kernel, iterations=1)
        cv2.putText(resultado, "3. Dilatacion", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    elif modo_filtro == '4':
        # Filtro Scharr: Mejor que Sobel para gradientes finos
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        scharr_x = cv2.Scharr(gris, cv2.CV_64F, 1, 0)
        resultado = cv2.convertScaleAbs(scharr_x) # Convertimos a 8 bits
        cv2.putText(resultado, "4. Filtro Scharr (X)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    elif modo_filtro == '5':
        # Filtro Relieve (Emboss): Convolución con matriz personalizada
        kernel_emboss = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]])
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resultado = cv2.filter2D(gris, -1, kernel_emboss)
        cv2.putText(resultado, "5. Filtro Relieve", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # --- 4. MOSTRAR RESULTADOS Y CAPTURAR TECLADO ---
    cv2.imshow('Camara - Procesamiento de Imagenes', resultado)
    mostrar_histograma(resultado, "Histograma en Tiempo Real")

    tecla = cv2.waitKey(1) & 0xFF
    if tecla == ord('q'):
        break
    elif chr(tecla) in ['1', '2', '3', '4', '5']:
        modo_filtro = chr(tecla)

cap.release()
cv2.destroyAllWindows()