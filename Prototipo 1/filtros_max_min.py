import cv2
import numpy as np

img = cv2.imread('esnupi_noise.png', 0) #0 para leer en escala de grises
if img is None:
    raise FileNotFoundError("No se pudo cargar la imagen. Verifica la ruta y el archivo.")
# cv2.imread('esnupi2.png', 0)
fil, col = img.shape

w_max = np.zeros((fil + 2, col + 2), dtype=np.uint8)
w_min = (np.ones((fil + 2, col + 2), dtype=np.uint8)) * 255
img_max = np.zeros((fil, col), dtype=np.uint8)
img_min = (np.zeros((fil, col), dtype=np.uint8)) * 255

for j in range(1, fil + 1):
    for i in range(1, col + 1):
        w_max[j, i] = img[j - 1, i - 1]
        w_min[j, i] = img[j - 1, i - 1]
        
        # Variables para w_max
        a = w_max[j - 1, i - 1]
        b = w_max[j, i - 1]
        c = w_max[j + 1, i - 1]
        d = w_max[j - 1, i]
        e = w_max[j, i]
        f = w_max[j + 1, i]
        g = w_max[j - 1, i + 1]
        h = w_max[j, i + 1]
        k = w_max[j + 1, i + 1]
        
        # Variables para w_min
        a1 = w_min[j - 1, i - 1]
        b1 = w_min[j, i - 1]
        c1 = w_min[j + 1, i - 1]
        d1 = w_min[j - 1, i]
        e1 = w_min[j, i]
        f1 = w_min[j + 1, i]
        g1 = w_min[j - 1, i + 1]
        h1 = w_min[j, i + 1]
        k1 = w_min[j + 1, i + 1]
        
        A_max = [[a, b, c, d, e, f, g, h, k]]
        A_min = [[a1, b1, c1, d1, e1, f1, g1, h1, k1]]
        
        maximo = np.amax(A_max)
        minimo = np.amin(A_min)
        
        img_max[j - 1, i - 1] = maximo
        img_min[j - 1, i - 1] = minimo
        
cv2.imshow('Imagen original', img)
cv2.imshow('Filtro maximo', img_max)
cv2.imshow('Filtro minimo', img_min)

cv2.waitKey(0)
cv2.destroyAllWindows()