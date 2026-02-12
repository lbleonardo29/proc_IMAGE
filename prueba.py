import numpy as np
import matplotlib.pyplot as plt

# Creamos una matriz de 256x256 donde cada fila es un valor de 0 a 255
gradiente = np.tile(np.arange(256, dtype="uint8"), (256, 1))

plt.imshow(gradiente, cmap='gray')
plt.title("Ejemplo 1: Gradiente Lineal (0-255)")
plt.colorbar(label='Intensidad del Píxel')
plt.show()