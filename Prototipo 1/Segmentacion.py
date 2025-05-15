import Messages as msg
import numpy as np
import cv2

class Segmentacion:
    def calcParticiones(self, ancho, alto, max_divisiones = 10):
        mejorError = float('inf') #El mejor error inicia en infinito
        mejorParticion = (1, 1) #La mejor partición. Inicial es de 1x1 píxel.
        
        for filas in range(1, max_divisiones+1):
            for cols in range(1, max_divisiones+1):
                anchoBloque = ancho / cols
                altoBloque = alto / filas
                aspecto = anchoBloque / altoBloque

                error = abs(aspecto - 1) #Medida de error (qué tan cuadrado es el bloque).

                if error < mejorError:
                    mejorError = error
                    mejorParticion = (filas, cols)
        
        return mejorParticion

    def umbraladoSegmentacion(self, img, maxSeg):
        try:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if maxSeg == 0:
                msg.alerta_message("Ajuste el número máximo de segmentos en Ajustar Constantes. Por lo mientras, el número ha sido ajustado a 10.")
                maxSeg = 10

            alt, anch = img.shape[:2]
            
            filas, cols = self.calcParticiones(anch, alt, max_divisiones=maxSeg)
            msg.alerta_message("La imagen se ha partido en segmentos de "+str(filas)+"x"+str(cols))
            
            resultado = np.zeros_like(img)

            for i in range(filas):
                for j in range(cols):
                    y_i = int(i * alt / filas)
                    y_f = int((i+1) * alt / filas)
                    x_i = int(j * anch / cols)
                    x_f = int((j+1) * anch / cols)
                    
                    bloque = img[y_i:y_f, x_i:x_f]
                    _, bloqueUmbralizado = cv2.threshold(bloque, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    resultado[y_i:y_f, x_i:x_f] = bloqueUmbralizado

            return resultado
        except Exception as e:
            msg.error_message(f"Ha ocurrido un error al aplicar segmentación por umbralización adaptativa: {str(e)}")
            print(f"Segmentación por umbralización adaptativa error: {str(e)}")
            return None

    def umbralizacionAdaptativa(self, img, kernel = 3, c=0):
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        try:
            fil, col = img.shape
            borde = kernel // 2
            img_extendida = np.pad(img, borde, mode='reflect')
            img_umbralizada = np.zeros_like(img, dtype=np.uint8)

            for j in range(fil):
                for i in range(col):
                    ventana = img_extendida[j:j+kernel, i:i+kernel]
                    media = np.mean(ventana)
                    valor_Umbral = media - c
                    img_umbralizada[j, i] = 255 if img[j, i] > valor_Umbral else 0

        except Exception as e:
            msg.error_message(f"Error al umbralizar la imagen: {str(e)}")
            print(f"Error en el umbralizado adaptativo de la imagen: {str(e)}")
            return None

        return img_umbralizada