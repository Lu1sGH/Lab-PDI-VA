# Importar librerías
import cv2 # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt # type: ignore

def abrirImg(nom):
    # ----- 1. Cargar y binarizar la imagen -----
    # Cargar imagen en escala de grises
    image = cv2.imread(nom, cv2.IMREAD_GRAYSCALE)

    # Verificar si la imagen fue cargada correctamente
    if image is None:
        print("Error al cargar la imagen.")
        exit()

    return image

def umbralizar(image, umbral):
    # Umbralización para binarizar la imagen
    _, binary_image = cv2.threshold(image, umbral, 255, cv2.THRESH_BINARY)
    return binary_image

def mostrarImg(image, titulo, cmap, isBar = False):
    # Mostrar imagen
    plt.figure(figsize=(6, 6))
    plt.imshow(image) if cmap == None else plt.imshow(image, cmap) # Por si no es necesario el
    cmap
    plt.title(titulo) # Título de la ventana

    if isBar: plt.colorbar() # Por si se va a mostrar la gráfica de colores
    plt.axis('off')
    plt.show()
    
def etiquetado(bin, conectividad):
    # ----- 2. Etiquetado de componentes conexas -----
    num_labels_n, labels_n = cv2.connectedComponents(bin, connectivity=conectividad)
    return num_labels_n, labels_n

def mostrarInfo(nl4, nl8):
    # Mostrar diferencias obtenidas entre Vecindad-4 y Vecindad-8
    print(f"Número de objetos detectados con vecindad-4: {nl4 - 1}")
    print(f"Número de objetos detectados con vecindad-8: {nl8 - 1}")
    # ----- 4. Comparación entre vecindad-4 y vecindad-8 -----
    diferencia = abs(nl4 - nl8)
    print(f"Diferencia entre vecindad-4 y vecindad-8: {diferencia}")

def dibujarContornos(binary_image):
    # ----- 3. Dibujar contornos y numerar los objetos -----
    # Convertir imagen binaria a imagen en color para dibujar contornos
    image_color = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
    # Encontrar contornos en la imagen
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Dibujar los contornos y numerar los objetos
    for i, contour in enumerate(contours):
        # Dibujar contorno (color verde)
        cv2.drawContours(image_color, [contour], -1, (0, 255, 0), 2)
        # Encontrar el centro del objeto y colocar el número
        x, y, w, h = cv2.boundingRect(contour)
        cv2.putText(image_color, f'{i + 1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return image_color
    
if __name__ == "__main__":
    umbral = 127

    img = abrirImg("temp14.jpg")
    binary = umbralizar(img, umbral)
    mostrarImg(binary, "Imagen binarizada", "gray")
    num_labels_4, labels_4 = etiquetado(binary, 4) # Vecindad-4
    num_labels_8, labels_8 = etiquetado(binary, 8) # Vecindad-8
    mostrarImg(labels_4, "Etiquetado con Vecindad-4", "jet", True)
    mostrarImg(labels_8, "Etiquetado con Vecindad-8", "jet", True)
    img_contornos = dibujarContornos(binary)
    mostrarImg(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB), "Objetos Detectados y Numerados", None)
    mostrarInfo(num_labels_4, num_labels_8)