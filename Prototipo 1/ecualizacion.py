import cv2
import numpy as np
import matplotlib.pyplot as plt
import Messages as msg

class Ecualizador:
    def ecualizar_uniformemente(self, imagen):
        try:
            if(len(imagen.shape) == 3):
                msg.alerta_message("El método no admite imágenes a color.")
                return imagen

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

    #Otras ecualizaciones

    def correccionGamma(self, img, gamma = 1.0):
        try:
            if(len(img.shape) == 3):
                msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255
                            for i in np.arange(256)]).astype("uint8")
            return cv2.LUT(img, table)
        except Exception as e:
            msg.error_message(f"Error al aplicar la correccion gamma: {str(e)}")
            print(f"Error al aplicar la correccion gamma: {str(e)}")
            return None

    def rayleigh(self, img, scale=30):
        try:
            if(len(img.shape) == 3):
                msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_flat = img.flatten()
            rayleigh_cdf = 1 - np.exp(-((np.arange(256) / scale) ** 2) / 2)
            rayleigh_cdf = rayleigh_cdf / rayleigh_cdf[-1]  # Normalizar a [0,1]
            img_eq = np.interp(img_flat, np.linspace(0, 255, 256), rayleigh_cdf * 255)
            return img_eq.reshape(img.shape).astype(np.uint8)
        except Exception as e:
            msg.error_message(f"Error al aplicar Rayleigh: {str(e)}")
            print(f"Error al aplicar rayleigh: {str(e)}")
            return None
    
    def hipercubica(self, img, degree=4):
        try:
            if(len(img.shape) == 3):
                msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_flat = img.flatten()
            norm_img = img_flat / 255.0
            hypercubic_cdf = norm_img ** degree
            hypercubic_cdf = hypercubic_cdf / hypercubic_cdf.max()
            img_eq = hypercubic_cdf * 255
            return img_eq.reshape(img.shape).astype(np.uint8)
        except Exception as e:
            msg.error_message(f"Error al aplicar ec hipercubica: {str(e)}")
            print(f"Error al aplicar ec hipercubica: {str(e)}")
            return None
    
    def exponencial(self, img, scale=50):
        try:
            if(len(img.shape) == 3):
                msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_flat = img.flatten()
            exp_cdf = 1 - np.exp(-img_flat / scale)
            exp_cdf = exp_cdf / exp_cdf.max()
            img_eq = exp_cdf * 255
            return img_eq.reshape(img.shape).astype(np.uint8)
        except Exception as e:
            msg.error_message(f"Error al aplicar ec exponencial: {str(e)}")
            print(f"Error al aplicar ec exponencial: {str(e)}")
            return None
    
    def logHiperbolica(self, img, c=1):
        try:
            if(len(img.shape) == 3):
                msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_float = img.astype(np.float32) + 1  # Evitar log(0)
            img_eq = c * np.log(img_float)
            img_eq = img_eq / np.max(img_eq) * 255
            return img_eq.astype(np.uint8)
        except Exception as e:
            msg.error_message(f"Error al aplicar ec Log Hiperbólica: {str(e)}")
            print(f"Error al aplicar ec Log Hiperbólica: {str(e)}")
            return None

    def expansion(self, img, new_min=0, new_max=255):
        try:
            if(len(img.shape) == 3):
                msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_float = img.astype(np.float32)
            img_norm = (img_float - np.min(img_float)) / (np.max(img_float) - np.min(img_float))
            img_expanded = img_norm * (new_max - new_min) + new_min
            return img_expanded.astype(np.uint8)
        except Exception as e:
            msg.error_message(f"Error al aplicar expansión: {str(e)}")
            print(f"Error al aplicar expansión: {str(e)}")
            return None
    
    def contraccion(self, img, center=127, width=100):
        try:
            if(len(img.shape) == 3):
                msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            img_float = img.astype(np.float32)
            min_val = center - width // 2
            max_val = center + width // 2
            img_contracted = np.clip(img_float, min_val, max_val)
            img_norm = (img_contracted - min_val) / (max_val - min_val) * 255
            return img_norm.astype(np.uint8)
        except Exception as e:
            msg.error_message(f"Error al aplicar contracción: {str(e)}")
            print(f"Error al aplicar contracción: {str(e)}")
            return None