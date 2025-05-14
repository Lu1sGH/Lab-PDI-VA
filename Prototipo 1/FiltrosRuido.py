import cv2
import numpy as np
import Messages as msg

class FiltrosYRuido:
    def ruido_salPimienta(self, imagen, p=0.02):
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
        ruido = np.random.normal(media, desEs, img.shape).astype(np.uint8)
        img_ruidosa = cv2.add(img, ruido)
        return img_ruidosa
    
    def ruidoMultiplicativo(self, img, media=0, desEs=0.1):
        img_float = img.astype(np.float32) / 255.0 #Convertir a float32 y normalizar a [0, 1]
        ruido = np.random.normal(loc=media, scale=desEs, size=img.shape).astype(np.float32) #Generar ruido multiplicativo con la misma forma que la imagen
        img_ruidosa = img_float * (1 + ruido) #Aplicar el ruido multiplicativo
        img_ruidosa = np.clip(img_ruidosa, 0, 1) * 255 #Limitar a [0, 1] y convertir de nuevo a uint8
        return img_ruidosa.astype(np.uint8)

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
        
    #Otros filtros
    def filtro_promediador(self, img, ksize=3):
        return cv2.blur(img, (ksize, ksize))

    def filtro_promediador_pesado(self, img, ksize=3):
        return cv2.boxFilter(img, -1, (ksize, ksize), normalize=False)

    def filtro_mediana(self, img, ksize=3):
        return cv2.medianBlur(img, ksize)

    def filtro_bilateral(self, img, d=9, sigmaColor=75, sigmaSpace=75):
        return cv2.bilateralFilter(img, d, sigmaColor, sigmaSpace)

    def filtro_gaussiano(self, img, ksize=3, sigmaX=0):
        return cv2.GaussianBlur(img, (ksize, ksize), sigmaX)