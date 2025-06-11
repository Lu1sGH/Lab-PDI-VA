import cv2
import numpy as np
import matplotlib.pyplot as plt
import Messages as msg

class Filtros_Paso_Altas:
    def gaussian_blur(self, image, kernel_size=5, sigma=1.4):
        """Aplica un filtro gaussiano a la imagen para reducir el ruido."""
        # Aplica el desenfoque gaussiano con un kernel cuadrado de tamaño kernel_size y desviación estándar sigma
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)

    def sobel_filters(self,img):
        """Calcula el gradiente de intensidad para detectar bordes."""
        # Definición de los kernels de Sobel
        Kx = np.array([[-1, 0, 1],
                    [-2, 0, 2],
                    [-1, 0, 1]], dtype=np.float32)
        Ky = np.array([[1, 2, 1],
                    [0, 0, 0],
                    [-1, -2, -1]], dtype=np.float32)
        """"Kernels de Scharr (Alternativos a Sobel). Más precisos para bordes.
        Kx = np.array([[-3, 0, 3],
                    [-10, 0, 10],
                    [-3, 0, 3]], dtype=np.float32)
        Ky = np.array([[3, 10, 3],
                    [0, 0, 0],
                    [-3, -10, -3]], dtype=np.float32)
        """

        # Convoluciona la imagen con los kernels para obtener las derivadas
        Ix = cv2.filter2D(img, cv2.CV_32F, Kx)
        Iy = cv2.filter2D(img, cv2.CV_32F, Ky)
        G = cv2.magnitude(Ix, Iy) # Ángulo del gradiente (en grados) usando la función arcotangente
        theta = cv2.phase(Ix, Iy, angleInDegrees=True)
        return G, theta # Retorna la magnitud y dirección del gradiente

    def non_maximum_suppression(self, G, theta):
        """Aplica la supresión no máxima para eliminar píxeles que no son bordes y afinar los mismos."""
        M, N = G.shape # Obtiene el tamaño de la imagen
        Z = np.zeros((M,N), dtype=np.float32) # Inicializa la imagen de salida con ceros
        angle = theta % 180 # Reduce los ángulos a un rango entre 0 y 180 grados

        #Comparar cada píxel con sus vecinos en la dirección del gradiente para conservar solo los máximos locales (posibles bordes)
        for i in range(1, M-1):
            for j in range(1, N-1):
                q = 255
                r = 255

                # 0 grados
                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                    q = G[i, j+1]
                    r = G[i, j-1]
                # 45 grados
                elif (22.5 <= angle[i,j] < 67.5):
                    q = G[i+1, j-1]
                    r = G[i-1, j+1]
                # 90 grados
                elif (67.5 <= angle[i,j] < 112.5):
                    q = G[i+1, j]
                    r = G[i-1, j]
                # 135 grados
                elif (112.5 <= angle[i,j] < 157.5):
                    q = G[i-1, j-1]
                    r = G[i+1, j+1]

                # Si el píxel actual es mayor que sus vecinos en la dirección del gradiente
                if G[i,j] >= q and G[i,j] >= r:
                    Z[i,j] = G[i,j] # Se conserva como posible borde
                else:
                    Z[i,j] = 0 # Se descarta

        return Z

    def double_threshold_manual(self, img, lowThreshold, highThreshold):
        """Aplica un doble umbral a la imagen para separar los confiables de los dudosos."""
        res = np.zeros_like(img, dtype=np.uint8)
        strong = 255
        weak = 75

        # Encuentra posiciones donde el valor del píxel supera el umbral alto
        strong_i, strong_j = np.where(img >= highThreshold)
        # Encuentra posiciones donde el valor del píxel está entre los dos umbrales
        weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

        # Asigna los valores fuertes y débiles en la imagen resultado
        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return res, weak, strong

    def hysteresis(self, img, weak, strong=255):
        """Conecta los píxeles débiles a los fuertes."""
        M, N = img.shape
        # Conectar píxeles débiles a los fuertes si son vecinos, convirtiéndolos en bordes definitivos.
        for i in range(1, M-1):
            for j in range(1, N-1):
                if img[i,j] == weak: # Si es un píxel débil
                    # Verifica si algún vecino es un píxel fuerte
                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                        img[i,j] = strong # Se conecta a un borde fuerte
                    else:
                        img[i,j] = 0
        return img

    def Canny(self, img, kernel = 5, sig = 1.4):
        """Aplica el filtro de Canny para detección de bordes."""
        try:
            if len(img.shape) == 3:
                msg.alerta_message("El método solo admite imágenes en escala de grises")
                return img

            blurred = self.gaussian_blur(img, kernel_size=kernel, sigma = sig)
            g, theta = self.sobel_filters(blurred)
            nms = self.non_maximum_suppression(g, theta)
            thresh, debil, fuerte = self.double_threshold_manual(nms, 70, 75)
            bordes = self.hysteresis(thresh, debil, strong = fuerte)

            return bordes
        except Exception as e:
            msg.error_message(f"Ha ocurrido un error al aplicar el filtro de Canny: {str(e)}")
            print(f"Canny error: {str(e)}")
            return None