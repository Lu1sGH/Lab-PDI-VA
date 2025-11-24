import Messages as msg
import numpy as np
import cv2

class Descriptores:
    def descFourier(self, img, precision=None):
        try:
            if len(img.shape) == 3:
                msg.alerta_message("La imagen se ha binarizado para su uso")
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            _, imgT = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            #Encontrar contornos
            contornos, _ = cv2.findContours(imgT, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            
            if len(contornos) == 0:
                raise ValueError("No se encontraron contornos en la imagen.")
            
            cont = contornos[0]
            
            if precision is not None and precision > len(cont):
                raise ValueError("El valor de precisión debe ser menor o igual al número de puntos del contorno.")
            
            contCplx = []

            if precision is not None:
                paso = len(cont) // precision
                if paso == 0:
                    paso = 1
                for i in range(0, len(cont), paso):
                    contCplx.append(cont[i, 0, 0] + 1j * cont[i, 0, 1])
            else:
                contCplx = cont[:, 0, 0] + 1j * cont[:, 0, 1]

            contCplx = contCplx - np.mean(contCplx)

            #Aplicar Transformada Rápida de Fourier
            descriptores = np.fft.fft(contCplx)
            descriptores = np.abs(descriptores) #Invarianza a la rotación
            descriptores[0] = 0  #Invarianza a la traslación
            
            #Invarianza a la escala
            epsilon = 1e-12
            id = np.where(np.abs(descriptores) < epsilon)[0]
            descriptores = descriptores / np.abs(descriptores[id[0]+1])

            return descriptores
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

    def momentosHU(self, img, verboose=True):
        try:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            momentos = cv2.moments(img)
            momHU = cv2.HuMoments(momentos)

            for i in range(len(momHU)):
                momHU[i] = -1 * np.sign(momHU[i]) * np.log10(abs(momHU[i])) if momHU[i] != 0 else 0

            txtmHU = ""
            for i in range(len(momHU)):
                txtmHU += f"Momento HU {i+1}: {momHU[i][0]} \n"
            
            if verboose:
                msg.todobien_message(str(txtmHU))
                print(f"Momentos:\n{momHU}")

            return momHU
        except Exception as e:
            msg.error_message(f"Error en la función Momentos HU: {str(e)}")
            print(f"Error en la función Momentos HU: {str(e)}")