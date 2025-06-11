import cv2
import numpy as np
import Messages as msg

class Ruido:
    def ruido_salPimienta(self, imagen, p=0.02):
        """Agrega ruido tipo sal y pimienta a una imagen."""
        img_ruidosa = imagen.copy() #Copia de la imagen original
        c = img_ruidosa.shape[2] if len(img_ruidosa.shape) == 3 else 1 #Verificamos si la imagen es a color o en escala de grises
        alt, anch = img_ruidosa.shape[:2] #Obtenemos las dimensiones de la imagen (alto y ancho)
        pixeles_ruidosos = int(alt * anch * p) #Calculo de la cantidad de pixeles ruidosos (píxeles a agregar ruido).

        for _ in range(pixeles_ruidosos): #Iteramos para agregar el ruido
            fil, col = np.random.randint(0, alt), np.random.randint(0, anch) #Obtenemos una posición aleatoria de la imagen
            if np.random.rand() < 0.5: #50% de probabilidad de agregar ruido tipo pimienta
                img_ruidosa[fil, col] = [0, 0, 0] if c == 3 else 0 #Si la imagen es a color
            else: #50% de probabilidad de agregar ruido tipo sal
                img_ruidosa[fil, col] = [255, 255, 255] if c == 3 else 255 #Si la imagen es a color
    
        return img_ruidosa
    
    def ruidoGaussiano(self, img, media=0, desEs=25):
        """Agrega ruido gaussiano a una imagen."""
        ruido = np.random.normal(media, desEs, img.shape).astype(np.uint8)
        img_ruidosa = cv2.add(img, ruido)
        return img_ruidosa
    
    def ruidoMultiplicativo(self, img, media=0, desEs=0.1):
        """Agrega ruido multiplicativo a una imagen."""
        img_float = img.astype(np.float32) / 255.0 #Convertir a float32 y normalizar a [0, 1]
        ruido = np.random.normal(loc=media, scale=desEs, size=img.shape).astype(np.float32) #Generar ruido multiplicativo con la misma forma que la imagen
        img_ruidosa = img_float * (1 + ruido) #Aplicar el ruido multiplicativo
        img_ruidosa = np.clip(img_ruidosa, 0, 1) * 255 #Limitar a [0, 1] y convertir de nuevo a uint8
        return img_ruidosa.astype(np.uint8)