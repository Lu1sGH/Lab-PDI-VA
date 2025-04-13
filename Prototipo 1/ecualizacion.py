import cv2
import numpy as np
import matplotlib.pyplot as plt

class Ecualizador:
    def __init__(self, ruta_imagen):
        self.imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
        if self.imagen is None:
            raise ValueError("No se pudo cargar la imagen.")
        self.imagen_ecualizada = None
        self.screen_width = 1920  # Ajustar según resolución de pantalla
        self.screen_height = 1080  # Ajustar según resolución de pantalla

    def center_window(self, window_name):
        window_size = cv2.getWindowImageRect(window_name) #(x, y, width, height)
        if window_size[2] > 0 and window_size[3] > 0:
            x = (self.screen_width - window_size[2]) // 2
            y = (self.screen_height - window_size[3]) // 2
            cv2.moveWindow(window_name, x, y)

    def mostrar_original(self):
        cv2.imshow("Imagen Original", self.imagen)
        self.center_window("Imagen Original")
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def mostrar_histograma_original(self):
        plt.hist(self.imagen.ravel(), bins=256, range=[0, 256], color='gray')
        plt.title("Histograma - Imagen Original")
        plt.xlabel("Intensidad")
        plt.ylabel("Frecuencia")
        plt.show()

    def ecualizar_uniformemente(self):
        g_min = 0
        g_max = 255
        total_pixeles = self.imagen.size
        hist = cv2.calcHist([self.imagen], [0], None, [256], [0, 256]).flatten() #la imagen, el canal (grises), no mascara, bins (rangos de intesidad), rango. De matriz a vector
        cdf = hist.cumsum() #suma acumulativa del histograma
        P_g = cdf / total_pixeles #probabilidad acumulada de cada nivel de gris
        F_g = np.round((g_max - g_min) * P_g + g_min).astype('uint8') #ecualizacion uniforme. valores a enteros sin signos
        self.imagen_ecualizada = F_g[self.imagen] #mapeo de la imagen original a la imagen ecualizada

    def mostrar_ecualizada(self):
        if self.imagen_ecualizada is not None:
            cv2.imshow("Imagen Ecualizada", self.imagen_ecualizada)
            self.center_window("Imagen Ecualizada")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def mostrar_histograma_ecualizada(self):
        if self.imagen_ecualizada is not None:
            plt.hist(self.imagen_ecualizada.ravel(), bins=256, range=[0, 256], color='blue')
            plt.title("Histograma - Imagen Ecualizada")
            plt.xlabel("Intensidad")
            plt.ylabel("Frecuencia")
            plt.show()