import cv2
import numpy as np
import matplotlib.pyplot as plt
import Messages as msg

class Ecualizador:
    def ecualizar_uniformemente(self, imagen):
        try:        
            g_min = 0
            g_max = 255
            total_pixeles = imagen.size
            hist = cv2.calcHist([imagen], [0], None, [256], [0, 256]).flatten() #la imagen, el canal (grises), no mascara, bins (rangos de intesidad), rango. De matriz a vector
            cdf = hist.cumsum() #suma acumulativa del histograma
            P_g = cdf / total_pixeles #probabilidad acumulada de cada nivel de gris
            F_g = np.round((g_max - g_min) * P_g + g_min).astype('uint8') #ecualizacion uniforme. valores a enteros sin signos
            imagen_ecualizada = F_g[imagen] #mapeo de la imagen original a la imagen ecualizada

            return imagen_ecualizada
        except Exception as e:
            msg.error_message(f"Error en la ecualización uniforme: {str(e)}")
            print(f"Error en la ecualización uniforme: {str(e)}")
            return None