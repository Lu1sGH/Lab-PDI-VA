import cv2
import numpy as np
import matplotlib.pyplot as plt

class Operaciones:
    def __init__(self):
        self.screen_width = 1920
        self.screen_height = 1080

    def center_window(self, window_name):
        window_size = cv2.getWindowImageRect(window_name)
        if window_size[2] > 0 and window_size[3] > 0:
            x = (self.screen_width - window_size[2]) // 2
            y = (self.screen_height - window_size[3]) // 2
            cv2.moveWindow(window_name, x, y)

    def aGris(self, imagen=None):
        return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    def umbralizar(self, image, umbral):
        # Umbralización para binarizar la imagen
        _, binary_image = cv2.threshold(image, umbral, 255, cv2.THRESH_BINARY)
        return binary_image

    def suma(self, valor=50, imagen=None):
        return cv2.add(imagen, valor)

    def resta(self, valor=50, imagen=None):
        return cv2.subtract(imagen, valor)

    def multiplicacion(self, factor=1.2, imagen=None):
        return cv2.multiply(imagen, factor)

    def esperar_cerrar(self):
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def operacion_and(self, ruta_imagen2):
        return self._operacion_logica(ruta_imagen2, 'AND')

    def operacion_or(self, ruta_imagen2):
        return self._operacion_logica(ruta_imagen2, 'OR')

    def operacion_xor(self, ruta_imagen2):
        return self._operacion_logica(ruta_imagen2, 'XOR')

    def _operacion_logica(self, img1, img2, tipo):
        img1 = cv2.resize(img1, (300, 300))
        img2 = cv2.resize(img2, (300, 300))

        if tipo == 'AND':
            return cv2.bitwise_and(img1, img2)
        elif tipo == 'OR':
            return cv2.bitwise_or(img1, img2)
        elif tipo == 'XOR':
            return cv2.bitwise_xor(img1, img2)
        else:
            raise ValueError("Operación lógica no válida.")

    def mostrar_histograma(self, imagen=None):
        plt.figure()
        plt.hist(imagen.ravel(), bins=256, range=[0, 256], color='gray')
        plt.title("Histograma - Imagen Original")
        plt.xlabel("Intensidad")
        plt.ylabel("Frecuencia")
        plt.show()

        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            hist = cv2.calcHist([imagen], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])

        plt.title('Histograma de la Imagen en Color')
        plt.show()

    def mostrar_componentes_RGB(self, imagen=None):
        b, g, r = cv2.split(imagen)

        zeros = np.zeros(imagen.shape[:2], dtype="uint8")
        cv2.imshow("Rojo", cv2.merge([zeros, zeros, r]))
        cv2.imshow("Verde", cv2.merge([zeros, g, zeros]))
        cv2.imshow("Azul", cv2.merge([b, zeros, zeros]))
