import cv2
import numpy as np
import matplotlib.pyplot as plt

def ejercicio1_filtro_filter2D():
    """
    Ejercicio 1: Aplicación de Filtro con cv2.filter2D().
    """

    try:
        img = cv2.imread(r'C:\Users\tanya\OneDrive\Escritorio\Procesamiento de imagenes\Trabajo Practico 2\317080.jpg', cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError("La imagen no se pudo cargar. Asegúrate de que la ruta sea correcta.")
    except FileNotFoundError as e:
        print(e)
        return

    # Visualizamos la imagen original
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title('Imagen Original')
    plt.axis('off')

    # Kernel del filtro 
    kernel_ej1 = np.array([
        [1, -2, 1],
        [-2, 4, -2],
        [1, -2, 1]
    ], dtype=np.float32)


    img_filtrada_ej1 = cv2.filter2D(img, -1, kernel_ej1, borderType=cv2.BORDER_REPLICATE)

    # Visualizamos la imagen filtrada para su análisis
    plt.subplot(1, 2, 2)
    plt.imshow(img_filtrada_ej1, cmap='gray')
    plt.title('Imagen aplicando el filtro 2D')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    ejercicio1_filtro_filter2D()
