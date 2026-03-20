import re

texto = """
--- OCR OUTPUT (PSM 3) ---
Reto Bingo

2/3/2026

Nombre de Jugador Leonardo Ballesteros L
"""

# Numero
match_num = re.search(r'Numero de Juego\s*[:\-]?\s*([a-zA-Z0-9]+)', texto, re.IGNORECASE)
print(f"Num match: {match_num.group(1) if match_num else 'None'}")

# Fecha
# En el texto anterior no dice "Fecha", solo dice "2/3/2026".
# Pero en la imagen hay una celda que dice "Fecha de Juego".
# Si Tesseract no lee "Fecha", la regex fallará.
# Vamos a probar una regex de fecha más general si la de fecha falla.
match_fecha = re.search(r'Fecha[^\d]*(\d{1,2}/\d{1,2}/\d{2,4})', texto, re.IGNORECASE)
if not match_fecha:
    match_fecha = re.search(r'(\d{1,2}/\d{1,2}/\d{2,4})', texto)
print(f"Fecha match: {match_fecha.group(1) if match_fecha else 'None'}")

# Nombre
match_nombre = re.search(r'Nombre de Jugador\s*[:\-]?\s*([A-Za-z\s]{3,})', texto, re.IGNORECASE)
print(f"Nombre match: {match_nombre.group(1).strip() if match_nombre else 'None'}")
