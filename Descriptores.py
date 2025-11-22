import Messages as msg
import numpy as np
import cv2

class Descriptores:
    def descFourier(self, img, precision=None):
        try:
            if len(img.shape) == 3:
                msg.alerta_message("La imagen se ha binarizado para su uso")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            _, imgT = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            
            #Encontrar contornos
            contornos, _ = cv2.findContours(imgT, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(contornos) == 0:
                raise ValueError("No se encontraron contornos en la imagen.")
            
            cont = contornos[0]
            
            if precision is not None and precision > len(cont):
                raise ValueError("El valor de precisión debe ser menor o igual al número de puntos del contorno.")
            
            contCplx = cont[:precision, 0, 0] + 1j * cont[:precision, 0, 1] if precision else cont[:, 0, 0] + 1j * cont[:, 0, 1]

            #Aplicar Transformada Rápida de Fourier
            descriotores = np.fft.fft(contCplx)
            descriotores [0] = 0 #Para que no importe la translación
            descriotores = descriotores / np.abs(descriotores[1]) #Para que no importe la escala

            return descriotores
        except Exception as e:
            msg.error_message(f"Error en la función Descriptor de Fourier: {str(e)}")
            print(f"Error en la función Descriptor de Fourier: {str(e)}")

    def reconstContDF(self, vectorCplx):
        try:
            w, h = 500, 500
            if vectorCplx is None:
                raise ValueError("No se tiene alamacenado ningún vector de descriptores de Fourier.")
            
            #Cálculo de la inversa de la Transformada Rápida de Fourier
            inversa = np.fft.ifft(vectorCplx)
            
            #Convertir a puntos 2D
            pts = np.vstack([inversa.real, inversa.imag]).T

            #Normalizar y centrar los puntos
            pts -= pts.min(axis=0) #Llevar a origen
            pts = pts / pts.max() #Normalizar [0,1]
            pts *= (h - 20) #Escalar para que quepa en la imagen
            pts = pts.astype(np.int32)

            #Crear imagen
            img = np.zeros((h, w, 3), dtype=np.uint8)
            cv2.polylines(img, [pts], True, (255, 255, 255), 2)

            return img
        except Exception as e:
            msg.error_message(f"Error en la función Descriptor de Fourier: {str(e)}")
            print(f"Error en la función Descriptor de Fourier: {str(e)}")