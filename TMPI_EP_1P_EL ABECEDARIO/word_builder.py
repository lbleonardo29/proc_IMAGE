import os
from datetime import datetime
from config import RUTA_SALIDA_TEXTO


class WordBuilder:
    """
    Gestiona la construcción de palabras y texto a partir de letras reconocidas.

    Estado interno:
        word_actual  → la palabra que se está construyendo ahora mismo
        text_final   → todas las palabras terminadas, separadas por espacios
        historial    → lista de todas las letras y eventos (para debug)

    Uso típico en main.py:
        wb = WordBuilder()
        wb.agregar_letra('H')
        wb.agregar_letra('O')
        wb.agregar_letra('L')
        wb.agregar_letra('A')
        wb.fin_palabra()         # 'word_actual' se vacía, 'HOLA' va a text_final
        wb.guardar_archivo()     # guarda "HOLA " en el .txt
    """

    def __init__(self):
        """Inicializa el constructor con estado vacío."""
        self.word_actual = ""
        self.text_final  = ""
        self.historial   = []  # lista de eventos para debug y reporte

    def agregar_letra(self, letra):
        """
        Agrega una letra a la palabra que se está construyendo.

        Args:
            letra: string de un carácter, por ejemplo 'A' o 'Ñ'
        """
        self.word_actual += letra
        self.historial.append(('LETRA', letra))
        print(f"  [WordBuilder] Letra: '{letra}' → Palabra actual: '{self.word_actual}'")

    def fin_palabra(self):
        """
        Cierra la palabra actual y la agrega al texto final.

        Si word_actual está vacío (el usuario dio fin de palabra sin
        haber dibujado ninguna letra), simplemente lo ignora.

        Flujo:
            word_actual = "HOLA"
            fin_palabra()
            → text_final = "HOLA "
            → word_actual = ""
        """
        if not self.word_actual:
            return  # nada que agregar

        self.text_final  += self.word_actual + " "
        self.historial.append(('FIN_PALABRA', self.word_actual))
        print(f"  [WordBuilder] Palabra completada: '{self.word_actual}'")
        print(f"  [WordBuilder] Texto acumulado: '{self.text_final}'")
        self.word_actual  = ""

    def fin_sesion(self):
        """
        Cierra la sesión: guarda cualquier palabra pendiente y
        genera el archivo de texto final.

        Si había una palabra en construcción cuando el usuario
        detuvo el sistema, la guarda antes de cerrar para no perderla.

        Returns:
            ruta del archivo guardado, o None si no había texto
        """
        if self.word_actual:
            print(f"  [WordBuilder] Guardando palabra pendiente: '{self.word_actual}'")
            self.fin_palabra()

        if not self.text_final.strip():
            print("  [WordBuilder] No hay texto para guardar.")
            return None

        return self.guardar_archivo()

    def guardar_archivo(self):
        """
        Guarda el texto final en un archivo .txt con timestamp.

        El nombre del archivo incluye la fecha y hora para evitar
        sobrescribir resultados de sesiones anteriores.

        Returns:
            ruta_archivo: string con la ruta donde se guardó el archivo
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        nombre_archivo = f"Ballesteros_{timestamp}.txt"
        ruta_archivo = os.path.join(
            os.path.dirname(RUTA_SALIDA_TEXTO),
            nombre_archivo
        ) if os.path.dirname(RUTA_SALIDA_TEXTO) else nombre_archivo

        contenido = self._generar_contenido_archivo(timestamp)

        with open(ruta_archivo, 'w', encoding='utf-8') as f:
            f.write(contenido)

        print(f"\n  [WordBuilder] Archivo guardado: {ruta_archivo}")
        return ruta_archivo

    def obtener_word_actual(self):
        """Devuelve la palabra que se está construyendo actualmente."""
        return self.word_actual

    def obtener_text_final(self):
        """Devuelve todo el texto acumulado de palabras terminadas."""
        return self.text_final

    def obtener_texto_completo(self):
        """
        Devuelve el texto completo: palabras terminadas + palabra actual.
        Útil para mostrar en la UI el estado completo en tiempo real.
        """
        return self.text_final + self.word_actual

    def esta_vacio(self):
        """Indica si no hay ningún texto acumulado todavía."""
        return not self.word_actual and not self.text_final

    def resetear(self):
        """Limpia todo el estado. Útil para empezar una nueva sesión."""
        self.word_actual = ""
        self.text_final  = ""
        self.historial   = []

    def _generar_contenido_archivo(self, timestamp):
        """
        Genera el contenido formateado del archivo de texto final.

        El archivo incluye un encabezado con metadatos y el texto generado,
        para que sea más útil como entregable del proyecto.

        Args:
            timestamp: string con la fecha/hora de la sesión

        Returns:
            string con el contenido completo del archivo
        """
        lineas = [
            "=" * 50,
            "  SISTEMA DE VISIÓN ARTIFICIAL — ABECEDARIO",
            f"  Sesión: {timestamp}",
            "=" * 50,
            "",
            "TEXTO DETECTADO:",
            "-" * 50,
            self.text_final.strip(),
            "",
            "-" * 50,
            f"Total de palabras: {len(self.text_final.split())}",
            f"Total de letras:   {len(self.text_final.replace(' ', ''))}",
            "",
            "HISTORIAL DE EVENTOS:",
            "-" * 50,
        ]

        for tipo, valor in self.historial:
            lineas.append(f"  [{tipo}] {valor}")

        lineas.append("=" * 50)

        return "\n".join(lineas)