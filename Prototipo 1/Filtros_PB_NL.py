import cv2
import numpy as np
import Messages as msg

class Filtros_PasoBajas_NoLineales:
    def filtro_max(self, img=None): #Implementación del filtro máximo
        if img is None: return None

        self.img = img
        self.fil, self.col = self.img.shape
        self.w_max = np.zeros((self.fil + 2, self.col + 2), dtype=np.uint8) #Se inicializa la máscara de 3x3 con ceros
        self.img_max = np.zeros((self.fil, self.col), dtype=np.uint8) #Se inicializa la imagen de salida con ceros

        for j in range(1, self.fil + 1):
            for i in range(1, self.col + 1):
                self.w_max[j, i] = self.img[j - 1, i - 1]

                # Variables para w_max (máscara de 3x3)
                a = self.w_max[j - 1, i - 1]
                b = self.w_max[j, i - 1]
                c = self.w_max[j + 1, i - 1]
                d = self.w_max[j - 1, i]
                e = self.w_max[j, i]
                f = self.w_max[j + 1, i]
                g = self.w_max[j - 1, i + 1]
                h = self.w_max[j, i + 1]
                k = self.w_max[j + 1, i + 1]

                A_max = [[a, b, c, d, e, f, g, h, k]]

                maximo = np.amax(A_max)

                self.img_max[j - 1, i - 1] = maximo

    
    def filtro_min(self, img=None): #Implementación del filtro mínimo
        if img is None: return None

        self.img = img
        self.fil, self.col = self.img.shape
        self.w_min = np.ones((self.fil + 2, self.col + 2), dtype=np.uint8) * 255 #Se inicializa la máscara de 3x3 con 255 (blanco)
        self.img_min = np.ones((self.fil, self.col), dtype=np.uint8) * 255 #Se inicializa la imagen de salida con 255 (blanco)

        for j in range(1, self.fil + 1):
            for i in range(1, self.col + 1):
                self.w_min[j, i] = self.img[j - 1, i - 1]

                # Variables para w_min (máscara de 3x3)
                a1 = self.w_min[j - 1, i - 1]
                b1 = self.w_min[j, i - 1]
                c1 = self.w_min[j + 1, i - 1]
                d1 = self.w_min[j - 1, i]
                e1 = self.w_min[j, i]
                f1 = self.w_min[j + 1, i]
                g1 = self.w_min[j - 1, i + 1]
                h1 = self.w_min[j, i + 1]
                k1 = self.w_min[j + 1, i + 1]

                A_min = [[a1, b1, c1, d1, e1, f1, g1, h1, k1]]

                minimo = np.amin(A_min)

                self.img_min[j - 1, i - 1] = minimo

    def aplicar_filtro(self, img, tipo_filtro):
        try:
            if len(img.shape) == 3 and img.shape[2] == 3:  # Verifica si la imagen es a color (3 canales)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Convierte a escala de grises

            if tipo_filtro == "Filtro Máximo":
                self.filtro_max(img)
                msg.todobien_message("Filtro máximo aplicado correctamente.")
                return self.img_max
            elif tipo_filtro == "Filtro Mínimo":
                self.filtro_min(img)
                msg.todobien_message("Filtro mínimo aplicado correctamente.")
                return self.img_min
        except Exception as e:
            msg.error_message(f"Error al aplicar el {tipo_filtro}: {str(e)}")
            print(f"Error al aplicar el {tipo_filtro}: {e}")
            return None
        
    #=====================OTROS FILTROS=====================

    def filtro_promediador(self, img, ksize=3):
        return cv2.blur(img, (ksize, ksize))

    def filtro_promediador_pesado(self, img):
        kernel = np.array([[1,1,1],[1,5,1],[1,1,1]]) / 13
        filtrada = cv2.filter2D(img, -1, kernel=kernel)
        return filtrada

    def filtro_mediana(self, img, ksize=3):
        return cv2.medianBlur(img, ksize)

    def filtro_bilateral(self, img, d=9, sigmaColor=75, sigmaSpace=75):
        return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

    def filtro_gaussiano(self, img, ksize=3, sigmaX=1):
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX)