import cv2
import numpy as np
import matplotlib.pyplot as plt
import Messages as msg

class Conteo:
    def conteoCompleto(self, img):
        try:
            umbral = 127

            binary = self.umbralizar(img, umbral)
            num_labels_4, labels_4 = self.etiquetado(binary, 4) # Vecindad-4
            num_labels_8, labels_8 = self.etiquetado(binary, 8) # Vecindad-8
            self.mostrarImg(labels_4, "Etiquetado con Vecindad-4", "jet", True)
            self.mostrarImg(labels_8, "Etiquetado con Vecindad-8", "jet", True)
            img_contornos = self.dibujarContornos(binary)
            self.mostrarImg(cv2.cvtColor(img_contornos, cv2.COLOR_BGR2RGB), "Objetos Detectados y Numerados", None)
            self.mostrarInfo(num_labels_4, num_labels_8)
        except Exception as e:
            msg.error_message(f"Error al realizar el conteo de objetos: {e}")
            print(f"Error al realizar el conteo de objetos: {e}")

    def umbralizar(self, image, umbral):
        # Umbralización para binarizar la imagen
        if(len(image.shape) == 3):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        _, binary_image = cv2.threshold(image, umbral, 255, cv2.THRESH_BINARY)
        return binary_image

    def mostrarImg(self, image, titulo, cmap, isBar = False):
        # Mostrar imagen
        plt.figure(figsize=(6, 6))
        plt.imshow(image) if cmap == None else plt.imshow(image, cmap) # Por si no es necesario el cmap
        plt.title(titulo) # Título de la ventana

        if isBar: plt.colorbar() # Por si se va a mostrar la gráfica de colores
        plt.axis('off')
        plt.show()
        
    def etiquetado(self, bin, conectividad):
        # ----- 2. Etiquetado de componentes conexas -----
        num_labels_n, labels_n = cv2.connectedComponents(bin, connectivity=conectividad)
        return num_labels_n, labels_n

    def dibujarContornos(self, binary_image):
        # ----- 3. Dibujar contornos y numerar los objetos -----
        # Convertir imagen binaria a imagen en color para dibujar contornos
        image_color = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)
        # Encontrar contornos en la imagen
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #print("\n--- Información de objetos detectados ---")
        for i, contour in enumerate(contours):
            # Dibujar contorno (color verde)
            cv2.drawContours(image_color, [contour], -1, (0, 255, 0), 2)
            # Coordenadas del objeto
            x, y, w, h = cv2.boundingRect(contour)

            # Calcular área y perímetro
            area = cv2.contourArea(contour)
            perimetro = cv2.arcLength(contour, True)

            # Mostrar área y perímetro en consola
            #print(f"Objeto {i + 1}: Área = {area:.2f} px², Perímetro = {perimetro:.2f} px")

            # Mostrar información en la imagen
            cv2.putText(image_color, f'{i+1}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            cv2.putText(image_color, f'A:{int(area)} px^2, P:{int(perimetro)} px', 
                        (x, y + h + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,0), 1)

        return image_color
    
    def mostrarInfo(self, nl4, nl8):
        # Mostrar diferencias obtenidas entre Vecindad-4 y Vecindad-8
        msg.todobien_message(f"Número de objetos detectados con vecindad-4: {nl4 - 1}")
        msg.todobien_message(f"Número de objetos detectados con vecindad-8: {nl8 - 1}")
        # ----- 4. Comparación entre vecindad-4 y vecindad-8 -----
        diferencia = abs(nl4 - nl8)
        msg.todobien_message(f"Diferencia entre vecindad-4 y vecindad-8: {diferencia}")