import cv2
import numpy as np
import os

class FiltrosMaxMin:
    def __init__(self, image_path):
        print("Directorio actual:", os.getcwd())
        self.image_path = image_path
        self.img = self.cargar_imagen()
        self.fil, self.col = self.img.shape
        self.w_max = np.zeros((self.fil + 2, self.col + 2), dtype=np.uint8)
        self.w_min = np.ones((self.fil + 2, self.col + 2), dtype=np.uint8) * 255
        self.img_max = np.zeros((self.fil, self.col), dtype=np.uint8)
        self.img_min = np.ones((self.fil, self.col), dtype=np.uint8) * 255

    def cargar_imagen(self):
        img = cv2.imread(self.image_path, 0)  # 0 para leer en escala de grises
        if img is None:
            raise FileNotFoundError(f"No se pudo cargar la imagen: {self.image_path}")
        return img

    def aplicar_filtros(self):
        for j in range(1, self.fil + 1):
            for i in range(1, self.col + 1):
                self.w_max[j, i] = self.img[j - 1, i - 1]
                self.w_min[j, i] = self.img[j - 1, i - 1]

                # Variables para w_max
                a = self.w_max[j - 1, i - 1]
                b = self.w_max[j, i - 1]
                c = self.w_max[j + 1, i - 1]
                d = self.w_max[j - 1, i]
                e = self.w_max[j, i]
                f = self.w_max[j + 1, i]
                g = self.w_max[j - 1, i + 1]
                h = self.w_max[j, i + 1]
                k = self.w_max[j + 1, i + 1]

                # Variables para w_min
                a1 = self.w_min[j - 1, i - 1]
                b1 = self.w_min[j, i - 1]
                c1 = self.w_min[j + 1, i - 1]
                d1 = self.w_min[j - 1, i]
                e1 = self.w_min[j, i]
                f1 = self.w_min[j + 1, i]
                g1 = self.w_min[j - 1, i + 1]
                h1 = self.w_min[j, i + 1]
                k1 = self.w_min[j + 1, i + 1]

                A_max = [[a, b, c, d, e, f, g, h, k]]
                A_min = [[a1, b1, c1, d1, e1, f1, g1, h1, k1]]

                maximo = np.amax(A_max)
                minimo = np.amin(A_min)

                self.img_max[j - 1, i - 1] = maximo
                self.img_min[j - 1, i - 1] = minimo

    def mostrar_resultados(self):
        cv2.imshow('Imagen original', self.img)
        cv2.imshow('Filtro máximo', self.img_max)
        cv2.imshow('Filtro mínimo', self.img_min)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
