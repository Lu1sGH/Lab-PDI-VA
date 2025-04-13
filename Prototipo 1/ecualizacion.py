import cv2
import numpy as np
import matplotlib.pyplot as plt

class Ecualizador:
    def __init__(self):
        self.screen_width = 1920  # Ajustar según resolución de pantalla
        self.screen_height = 1080  # Ajustar según resolución de pantalla

    def center_window(self, window_name):
        window_size = cv2.getWindowImageRect(window_name) #(x, y, width, height)
        if window_size[2] > 0 and window_size[3] > 0:
            x = (self.screen_width - window_size[2]) // 2
            y = (self.screen_height - window_size[3]) // 2
            cv2.moveWindow(window_name, x, y)

    def ecualizar_uniformemente(self, imagen):
        g_min = 0
        g_max = 255
        total_pixeles = imagen.size
        hist = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten() #la imagen, el canal (grises), no mascara, bins (rangos de intesidad), rango. De matriz a vector
        cdf = hist.cumsum() #suma acumulativa del histograma
        P_g = cdf / total_pixeles #probabilidad acumulada de cada nivel de gris
        F_g = np.round((g_max - g_min) * P_g + g_min).astype('uint8') #ecualizacion uniforme. valores a enteros sin signos
        imagen_ecualizada = F_g[imagen] #mapeo de la imagen original a la imagen ecualizada

        return imagen_ecualizada