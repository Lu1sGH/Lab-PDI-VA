import cv2
import numpy as np
import matplotlib.pyplot as plt
import Messages as msg

class Operaciones:
    def aGris(self, imagen=None):
        try:
            if len(imagen.shape) == 2:
                return imagen
            
            return cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            msg.error_message(f"Error al convertir a escala de grises: {str(e)}")
            print(f"Error al convertir a escala de grises: {str(e)}")
            return None
    
    def umbralizar(self, image, umbral):
        try:
            #Umbralización para binarizar la imagen
            _, binary_image = cv2.threshold(image, umbral, 255, cv2.THRESH_BINARY)
            return binary_image
        except Exception as e:
            msg.error_message(f"Error al umbralizar la imagen: {str(e)}")
            print(f"Error al umbralizar la imagen: {str(e)}")
            return None

    def suma(self, img1=None, img2 = 50):
        try:
            if type(img2) is not int and type(img2) is not float: #Si la imagen 2 no es un número, sino una imagen
                fil, col = img1.shape[:2]
                fil2, col2 = img2.shape[:2]

                if fil != fil2 or col != col2:
                    img2 = cv2.resize(img2, (col, fil))

            return cv2.add(img1, img2)
        except Exception as e:
            msg.error_message(f"Error al sumar valor a la imagen: {str(e)}")
            print(f"Error al sumar valor a la imagen: {str(e)}")
            return None

    def resta(self, img1=None, img2 = 50):
        try:
            if type(img2) is not int and type(img2) is not float: #Si la imagen 2 no es un número, sino una imagen
                fil, col = img1.shape[:2]
                fil2, col2 = img2.shape[:2]

                if fil != fil2 or col != col2:
                    img2 = cv2.resize(img2, (col, fil))

            return cv2.subtract(img1, img2)
        except Exception as e:
            msg.error_message(f"Error al restar valor a la imagen: {str(e)}")
            print(f"Error al restar valor a la imagen: {str(e)}")
            return None

    def multiplicacion(self, img1=None, img2 = 1.2):
        try:
            if type(img2) is not int and type(img2) is not float: #Si la imagen 2 no es un número, sino una imagen
                fil, col = img1.shape[:2]
                fil2, col2 = img2.shape[:2]

                if fil != fil2 or col != col2:
                    img2 = cv2.resize(img2, (col, fil))
            
            return cv2.multiply(img1, img2)
        except Exception as e:
            msg.error_message(f"Error al multiplicar la imagen: {str(e)}")
            print(f"Error al multiplicar la imagen: {str(e)}")
            return None

    def _operacion_logica(self, img1, img2, tipo):
        try: 
            fil, col = img1.shape[:2]
            fil2, col2 = img2.shape[:2]

            if fil != fil2 or col != col2:
                img2 = cv2.resize(img2, (col, fil))

            if tipo == 'AND':
                return cv2.bitwise_and(img1, img2)
            elif tipo == 'OR':
                return cv2.bitwise_or(img1, img2)
            elif tipo == 'XOR':
                return cv2.bitwise_xor(img1, img2)
        except Exception as e:
            msg.error_message(f"Error en la operación lógica {tipo}: {str(e)}")
            print(f"Error en la operación lógica: {str(e)}")
            return None
        
    def negacion(self, img):
        try:
            return cv2.bitwise_not(img)
        except Exception as e:
            msg.error_message(f"Error en la negación: {str(e)}")
            print(f"Error en bitwise_not: {str(e)}")
            return None


    def mostrar_histograma(self, imagen=None):
        try:
            #Mostrar histograma de la imagen en escala de grises
            plt.figure()  #Crea una nueva ventana para el histograma en escala de grises
            plt.hist(imagen.ravel(), bins=256, range=[0, 256], color='gray')
            plt.title("Histograma - Imagen Original")
            plt.xlabel("Intensidad")
            plt.ylabel("Frecuencia")
            plt.show(block=False)

            if len(imagen.shape) == 3 and imagen.shape[2] == 3:  #Verifica que la imagen tenga 3 canales (RGB)
                #Mostrar histograma de la imagen en color (para cada canal RGB)
                plt.figure()  #Crea una nueva ventana para el histograma de la imagen en color
                color = ('b', 'g', 'r')
                for i, col in enumerate(color):
                    hist = cv2.calcHist([imagen], [i], None, [256], [0, 256])
                    plt.plot(hist, color=col)
                    plt.xlim([0, 256])

                plt.title('Histograma de la Imagen en Color')
                plt.xlabel("Intensidad")
                plt.ylabel("Frecuencia")
                plt.show(block=False)
        except Exception as e:
            msg.error_message(f"Error al mostrar el histograma: {str(e)}")
            print(f"Error al mostrar el histograma: {str(e)}")

    def mostrar_componentes_RGB(self, imagen=None):
        try:
            b, g, r = cv2.split(imagen)

            zeros = np.zeros(imagen.shape[:2], dtype="uint8")
            cv2.imshow("Rojo", cv2.merge([zeros, zeros, r]))
            cv2.imshow("Verde", cv2.merge([zeros, g, zeros]))
            cv2.imshow("Azul", cv2.merge([b, zeros, zeros]))
        except Exception as e:
            msg.error_message(f"Error al mostrar los componentes RGB: {str(e)}")
            print(f"Error al mostrar los componentes RGB: {str(e)}")