import Messages as msg
import numpy as np
import cv2 as cv

class Mascaras_Operadores:
    def __init__(self):
        self.kernelsFC = [
            (1/(2*np.sqrt(2))) * np.array([[1,  np.sqrt(2),  1],
                                        [0,  0,          0],
                                        [-1, -np.sqrt(2), -1]]),
            (1/(2*np.sqrt(2))) * np.array([[1, 0, -1],
                                        [np.sqrt(2), 0, -np.sqrt(2)],
                                        [1, 0, -1]]),
            (1/(2*np.sqrt(2))) * np.array([[0, -1, np.sqrt(2)],
                                        [1, 0, -1],
                                        [-np.sqrt(2), 1, 0]]),
            (1/(2*np.sqrt(2))) * np.array([[np.sqrt(2), -1, 0],
                                        [-1, 0, 1],
                                        [0, 1, -np.sqrt(2)]]),
            (1/2) * np.array([[0, 1, 0],
                            [-1, 0, -1],
                            [0, 1, 0]]),
            (1/2) * np.array([[-1, 0, 1],
                            [0, 0, 0],
                            [1, 0, -1]]),
            (1/6) * np.array([[1, -2, 1],
                            [-2, 4, -2],
                            [1, -2, 1]]),
            (1/6) * np.array([[-2, 1, -2],
                            [1, 4, 1],
                            [-2, 1, -2]]),
            (1/3) * np.array([[1, 1, 1],
                            [1, 1, 1],
                            [1, 1, 1]])
        ]

        self.compass_kirsch = {
            "N": np.array([[ 5,  5,  5],
                        [-3,  0, -3],
                        [-3, -3, -3]], dtype=np.float32),

            "NE": np.array([[ 5,  5, -3],
                            [ 5,  0, -3],
                            [-3, -3, -3]], dtype=np.float32),

            "E": np.array([[ 5, -3, -3],
                        [ 5,  0, -3],
                        [ 5, -3, -3]], dtype=np.float32),

            "SE": np.array([[-3, -3, -3],
                            [ 5,  0, -3],
                            [ 5,  5, -3]], dtype=np.float32),

            "S": np.array([[-3, -3, -3],
                        [-3,  0, -3],
                        [ 5,  5,  5]], dtype=np.float32),

            "SW": np.array([[-3, -3, -3],
                            [-3,  0,  5],
                            [-3,  5,  5]], dtype=np.float32),

            "W": np.array([[-3, -3,  5],
                        [-3,  0,  5],
                        [-3, -3,  5]], dtype=np.float32),

            "NW": np.array([[-3,  5,  5],
                            [-3,  0,  5],
                            [-3, -3, -3]], dtype=np.float32)
        }
    
    def frei_chen(self, img):
        if len(img.shape) == 3:
            msg.alerta_message("La imagen se ha convertido a escala de grises.")
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        #Se aplica la convolucón con cada kernel
        results = [cv.filter2D(img.astype(np.float32), -1, k) for k in self.kernelsFC]

        #Para aplicar np.sqrt(M/S)
        #M = suma de cuadrados de la convolución de los primeros 4 kernels
        M = sum(r**2 for r in results[0:4])

        #S = suma de cuadrados de todas las convoluciones
        S = sum(r**2 for r in results)

        result = np.sqrt(M / (S + 1e-8))  #Añadir un pequeño valor para evitar división por cero

        return (result * 255).astype(np.uint8)

    def kirsch(self, img, dir = "T"):
        if len(img.shape) == 3:
            msg.alerta_message("La imagen se ha convertido a escala de grises.")
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        if dir == "T": #Todas las direcciones aplicadas
            results = [cv.filter2D(img.astype(np.float32), -1, k) for k in self.compass_kirsch.values()]
            result = np.maximum.reduce(results) #Se elige el valor máximo entre todas las direcciones
        else:
            if dir not in self.compass_kirsch: #Dirección inválida
                msg.error_message("Dirección inválida para el operador de Kirsch.")
                return None
            result = cv.filter2D(img.astype(np.float32), -1, self.compass_kirsch[dir])
        
        return cv.normalize(result, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    
    def sobel(self, img,):
        if len(img.shape) == 3:
            msg.alerta_message("La imagen se ha convertido a escala de grises.")
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype = np.float32)
        
        ky = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]], dtype = np.float32)
        
        gx = cv.filter2D(img.astype(np.float32), -1, kx)
        gy = cv.filter2D(img.astype(np.float32), -1, ky)

        #Magnitud del gradiente
        mag = np.hypot(gx, gy)
        return cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)
    
    def prewitt(self, img): 
        if len(img.shape) == 3:
            msg.alerta_message("La imagen se ha convertido a escala de grises.")
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        kx = np.array([[-1, 0, 1],
                       [-1, 0, 1],
                       [-1, 0, 1]], dtype = np.float32)
        
        ky = np.array([[1, 1, 1],
                       [0, 0, 0],
                       [-1, -1, -1]], dtype = np.float32)
        
        gx = cv.filter2D(img.astype(np.float32), -1, kx)
        gy = cv.filter2D(img.astype(np.float32), -1, ky)

        mag = np.hypot(gx, gy)
        return cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    def roberts (self, img): 
        if len(img.shape) == 3:
            msg.alerta_message("La imagen se ha convertido a escala de grises.")
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        kx = np.array([[1, 0],
                      [0, -1]], dtype= np.float32)
        
        ky = np.array([[0, 1],
                       [-1, 0]], dtype= np.float32)

        gx = cv.filter2D(img.astype(np.float32), -1, kx)
        gy = cv.filter2D(img.astype(np.float32), -1, ky) 

        mag = np.hypot(gx, gy)
        return cv.normalize(mag, None, 0, 225, cv.NORM_MINMAX).astype(np.uint8)
    