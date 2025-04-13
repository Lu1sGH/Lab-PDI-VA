import cv2
import numpy as np

class Operaciones:
    def __init__(self, ruta_imagen):
        self.imagen = cv2.imread(ruta_imagen)
        if self.imagen is None:
            raise ValueError("No se pudo cargar la imagen.")
        self.screen_width = 1920
        self.screen_height = 1080

    def center_window(self, window_name):
        window_size = cv2.getWindowImageRect(window_name)
        if window_size[2] > 0 and window_size[3] > 0:
            x = (self.screen_width - window_size[2]) // 2
            y = (self.screen_height - window_size[3]) // 2
            cv2.moveWindow(window_name, x, y)

    def operaciones_aritmeticas(self, valor=50, factor=1.2):
        suma = cv2.add(self.imagen, valor)
        resta = cv2.subtract(self.imagen, valor)
        multiplicacion = cv2.multiply(self.imagen, factor)

        cv2.imshow("Original", self.imagen)
        self.center_window("Original")

        cv2.imshow("Suma", suma)
        self.center_window("Suma")

        cv2.imshow("Resta", resta)
        self.center_window("Resta")

        cv2.imshow("Multiplicación", multiplicacion)
        self.center_window("Multiplicación")

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def operaciones_logicas(self, ruta_imagen2):
        img2 = cv2.imread(ruta_imagen2)
        if img2 is None:
            raise ValueError("No se pudo cargar la segunda imagen.")

        img1 = cv2.resize(self.imagen, (300, 300))
        img2 = cv2.resize(img2, (300, 300))

        and_img = cv2.bitwise_and(img1, img2)
        or_img = cv2.bitwise_or(img1, img2)
        xor_img = cv2.bitwise_xor(img1, img2)

        cv2.imshow("AND", and_img)
        self.center_window("AND")

        cv2.imshow("OR", or_img)
        self.center_window("OR")

        cv2.imshow("XOR", xor_img)
        self.center_window("XOR")

        cv2.waitKey(0)
        cv2.destroyAllWindows()