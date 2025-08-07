import cv2
import numpy as np
import matplotlib.pyplot as plt

def agregar_ruido_sal_pimienta(imagen, probabilidad=0.01):
    """
    Ejercicio 1: Aplicación de Filtro con cv2.filter2D()
    Agregamos ruido de sal y pimienta a una imagen.
    """
    imagen_ruido = np.copy(imagen)
    num_sal = np.ceil(probabilidad * imagen.size * 0.5).astype(int)
    coords_sal = [np.random.randint(0, i - 1, num_sal) for i in imagen.shape]
    imagen_ruido[coords_sal[0], coords_sal[1]] = 255
    
    num_pimienta = np.ceil(probabilidad * imagen.size * 0.5).astype(int)
    coords_pimienta = [np.random.randint(0, i - 1, num_pimienta) for i in imagen.shape]
    imagen_ruido[coords_pimienta[0], coords_pimienta[1]] = 0
    
    return imagen_ruido

def filtro_punto_medio_alfa_recortado(imagen, P, Q, T):
    """
    Implementamos el filtro de Punto Medio Alfa.

    """
    h, w = imagen.shape
    img_filtrada = np.zeros_like(imagen, dtype=np.uint8)
    padding_h = Q // 2
    padding_w = P // 2
    
    # Manejo de bordes
    img_padded = cv2.copyMakeBorder(imagen, padding_h, padding_h, padding_w, padding_w, cv2.BORDER_REPLICATE)
    
    for i in range(padding_h, h + padding_h):
        for j in range(padding_w, w + padding_w):
            ventana = img_padded[i-padding_h:i+padding_h+1, j-padding_w:j+padding_w+1]
            pixeles_ordenados = np.sort(ventana.flatten())
            pixeles_recortados = pixeles_ordenados[T:-T] if T > 0 else pixeles_ordenados
            
            if pixeles_recortados.size > 0:
                media = np.mean(pixeles_recortados)
                img_filtrada[i-padding_h, j-padding_w] = np.uint8(media)
            else:
                img_filtrada[i-padding_h, j-padding_w] = imagen[i-padding_h, j-padding_w]
            
    return img_filtrada

def ejercicio2_filtro_manual():
    """
    Implementamos el manual de filtro.
    """


    try:
        img_original = cv2.imread(r'C:\Users\tanya\OneDrive\Escritorio\Procesamiento de imagenes\Trabajo Practico 2\317080.jpg', cv2.IMREAD_GRAYSCALE)
        if img_original is None:
            raise FileNotFoundError("La imagen no se pudo cargar. Asegúrate de que la ruta sea correcta.")
    except FileNotFoundError as e:
        print(e)
        return

    # Agregamos ruido de sal y pimienta a la imagen original.
    img_con_ruido = agregar_ruido_sal_pimienta(img_original, probabilidad=0.05)

    P, Q, T = 3, 3, 1
    
    # Aplicamos el filtro punto medio alfa
    img_filtrada_manual = filtro_punto_medio_alfa_recortado(img_con_ruido, P, Q, T)

    # Aplicamos un filtro promedio 3x3 usando cv2.filter2D() para la comparación
    kernel_prom = np.ones((3, 3), np.float32) / 9
    img_filtrada_promedio = cv2.filter2D(img_con_ruido, -1, kernel_prom)
    
    # Visualizamos los resultados para el análisis
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img_con_ruido, cmap='gray')
    plt.title('Imagen con Ruido')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(img_filtrada_manual, cmap='gray')
    plt.title('Filtro de la Diapositiva 36 (manual)')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(img_filtrada_promedio, cmap='gray')
    plt.title('Filtro Promedio (filter2D) para comparación')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    print("\nAnálisis visual para el informe:")

if __name__ == "__main__":
    ejercicio2_filtro_manual()
