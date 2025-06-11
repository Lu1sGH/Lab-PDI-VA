import Messages as msg
import numpy as np
import cv2

class Segmentacion:
    def calcParticiones(self, ancho, alto, max_divisiones = 10):
        """Algoritmo para calcular la mejor partición de una imagen en bloques cuadrados o rectangulares."""
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
        """Segmentación de una imagen por umbralización adaptativa por partición."""
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
        """Aplica un umbral adaptativo a la imagen por propiedades locales."""
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
    
    #OTRAS SEGMENTACIONES

    def segmentacionUmbralMedia(self, img):
        """Segmentación completa por umbral de la media."""
        try:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            umbral = np.mean(img)
            _, img_segmentada = cv2.threshold(img, umbral, 255, cv2.THRESH_BINARY)
            return img_segmentada
        except Exception as e:
            msg.error_message(f"Error en segmentación por umbral media: {str(e)}")
            return None

    def segmentacionOtsu(self, img):
        """Segmentación completa por método de Otsu."""
        try:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, img_segmentada = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            return img_segmentada
        except Exception as e:
            msg.error_message(f"Error en segmentación Otsu: {str(e)}")
            return None

    def segmentacionMultiumbral(self, img, niveles=3):
        """Segmentación completa por multiumbralización."""
        try:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            hist = cv2.calcHist([img], [0], None, [256], [0,256]).ravel()
            total = img.size
            thresholds = []
            step = 256 // (niveles + 1)

            for i in range(1, niveles + 1):
                thresholds.append(i * step)

            img_segmentada = np.zeros_like(img)
            for i, t in enumerate(thresholds):
                img_segmentada[img > t] = int(255 * (i + 1) / (niveles + 1))

            return img_segmentada
        except Exception as e:
            msg.error_message(f"Error en segmentación multiumbral: {str(e)}")
            return None

    def segmentacionKapur(self, img):
        """Segmentación completa por entropía de Kapur."""
        try:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            hist = cv2.calcHist([img], [0], None, [256], [0, 256]).ravel()
            prob = hist / hist.sum()
            cumsum = np.cumsum(prob)

            entropy = np.zeros(256)
            for t in range(1, 255):
                p1 = prob[:t]
                p2 = prob[t:]

                w0 = cumsum[t]
                w1 = 1 - w0

                if w0 > 0 and w1 > 0:
                    h0 = -np.sum(p1[p1 > 0] * np.log(p1[p1 > 0])) / w0
                    h1 = -np.sum(p2[p2 > 0] * np.log(p2[p2 > 0])) / w1
                    entropy[t] = h0 + h1

            t_max = np.argmax(entropy)
            _, img_segmentada = cv2.threshold(img, t_max, 255, cv2.THRESH_BINARY)
            return img_segmentada
        except Exception as e:
            msg.error_message(f"Error en segmentación por entropía de Kapur: {str(e)}")
            return None

    def segmentacionUmbralBanda(self, img, t1=150, t2=200):
        """Segmentación mediante umbral banda [t1, t2]."""
        try:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_segmentada = cv2.inRange(img, t1, t2)
            return img_segmentada
        except Exception as e:
            msg.error_message(f"Error en segmentación por umbral banda: {str(e)}")
            return None

    def segmentacionMinimoHistograma(self, img):
        """Segmentación por mínimo del histograma."""
        try:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
            hist = cv2.calcHist([img], [0], None, [256], [0, 256])
            smooth_hist = cv2.GaussianBlur(hist, (5,1), 0)
            min_pos = np.argmin(smooth_hist[50:200]) + 50
            _, img_segmentada = cv2.threshold(img, min_pos, 255, cv2.THRESH_BINARY)
            return img_segmentada
        except Exception as e:
            msg.error_message(f"Error en segmentación por mínimo de histograma: {str(e)}")
            return None