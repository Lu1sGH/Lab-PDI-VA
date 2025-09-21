import Messages as msg
import numpy as np
import cv2 as cv

class Mascaras_Operadores:
    def __init__(self):
        pass

    import numpy as np

    def frei_chen(self, img):
        if len(img.shape) == 3:
            msg.alerta_message("La imagen se ha convertido a escala de grises.")
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        kernels = [
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

        #Se aplica la convolucón con cada kernel
        responses = [cv.filter2D(img.astype(np.float32), -1, k) for k in kernels]

        #Para aplicar np.sqrt(M/S)
        #M = suma de cuadrados de la convolución de los primeros 4 kernels
        M = sum(r**2 for r in responses[0:4])

        #S = suma de cuadrados de todas las convoluciones
        S = sum(r**2 for r in responses)

        result = np.sqrt(M / (S + 1e-8))  #Añadir un pequeño valor para evitar división por cero

        return (result * 255).astype(np.uint8)
