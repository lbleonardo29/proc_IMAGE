import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def test_ocr():
    img = cv2.imread('c:/Proyectos_TPIM/proc_IMAGE/TMPI_T2_4BINGO/bingo_tablero.png')
    if img is None:
        print("Image not found")
        return
    gris = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    procesada = cv2.adaptiveThreshold(gris, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 5)
    texto_completo = pytesseract.image_to_string(procesada, lang='eng', config='--psm 3')
    print("=== OCR OUTPUT ===")
    print(texto_completo[:1000])
    print("==================")

if __name__ == "__main__":
    test_ocr()
