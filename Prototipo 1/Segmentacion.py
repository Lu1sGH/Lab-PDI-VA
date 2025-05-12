import Messages as msg
import numpy as np

class Segmentacion:
    def umbralizacionAdaptativa(self, img, kernel = 3, c=0):
        if(len(img.shape) == 3):
            msg.alerta_message("El método no admite imágenes a color.")
            return None
        
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