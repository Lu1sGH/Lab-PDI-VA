import Messages as msg
import numpy as np
import cv2

class RecObjs:

    def genVec(self, img, descObj):
        try:
            if descObj is None:
                raise ValueError("No se ha proporcionado ningún descriptor de objeto.")
            
            vDF = descObj.descFourier(img, precision=10)
            vDF = vDF.astype(np.float64)
            momHu = descObj.momentosHU(img, verboose=False)
            vecgen = []

            for i in momHu:
                vecgen.append(float(i[0]))
            
            for i in vDF:
                vecgen.append(float(i))
            print(f"Vector: {vecgen}")
            print(f"Longitud del vector: {len(vecgen)}")
        except Exception as e:
            msg.error_message(f"Error en la función Generar Vectores: {str(e)}")
            print(f"Error en la función Generar Vectores: {str(e)}")