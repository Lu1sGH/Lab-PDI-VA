import cv2
import numpy as np
import Messages as msg

class Filtros_PasoBajas_NoLineales:
    def filtro_max(self, img, kernel=3):
        """Implementa un filtro máximo."""
        if img is None: return None

        self.img = img
        self.fil, self.col = self.img.shape
        pad = kernel // 2

        # Padding de la imagen para bordes
        padded_img = np.pad(self.img, pad_width=pad, mode='constant', constant_values=0)
        self.img_max = np.zeros_like(self.img, dtype=np.uint8)

        for j in range(self.fil):
            for i in range(self.col):
                region = padded_img[j:j+kernel, i:i+kernel]
                self.img_max[j, i] = np.max(region)


    def filtro_min(self, img, kernel=3):
        """Implementa un filtro mínimo."""
        if img is None: return None

        self.img = img
        self.fil, self.col = self.img.shape
        pad = kernel // 2 #Calcula el borde necesario para la imagen en funcion el tamaño del kernel

        # Padding de la imagen para bordes
        padded_img = np.pad(self.img, pad_width=pad, mode='constant', constant_values=255)
        self.img_min = np.zeros_like(self.img, dtype=np.uint8)

        for j in range(self.fil):
            for i in range(self.col):
                region = padded_img[j:j+kernel, i:i+kernel]
                self.img_min[j, i] = np.min(region)

    def aplicar_filtro(self, img, tipo_filtro, kernel):
        try:
            if len(img.shape) == 3 and img.shape[2] == 3:  # Verifica si la imagen es a color (3 canales)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convierte a escala de grises

            if tipo_filtro == "Filtro Máximo":
                self.filtro_max(img, kernel=kernel)
                msg.todobien_message("Filtro máximo aplicado correctamente.")
                return self.img_max
            elif tipo_filtro == "Filtro Mínimo":
                self.filtro_min(img, kernel=kernel)
                msg.todobien_message("Filtro mínimo aplicado correctamente.")
                return self.img_min
        except Exception as e:
            msg.error_message(f"Error al aplicar el {tipo_filtro}: {str(e)}")
            print(f"Error al aplicar el {tipo_filtro}: {e}")
            return None
        
    #=====================OTROS FILTROS=====================

    def filtro_promediador(self, img, ksize=3):
        return cv2.blur(img, (ksize, ksize))

    def filtro_promediador_pesado(self, img, N = 5):
        kernel = np.array([[1,1,1],[1,N,1],[1,1,1]]) / (8 + N)
        filtrada = cv2.filter2D(img, -1, kernel=kernel)
        return filtrada

    def filtro_mediana(self, img, ksize=3):
        return cv2.medianBlur(img, ksize)

    def filtro_bilateral(self, img, d=9, sigmaColor=75, sigmaSpace=75):
        return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

    def filtro_gaussiano(self, img, ksize=3, sigmaX=1):
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX)