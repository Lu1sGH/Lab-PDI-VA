import Messages as msg
import numpy as np
import cv2

class TemplateMatching:
    def tm_OpenCV(self, img, template, metodo=''):
        try:
            imgOri = img.copy()
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            metodos = {
                'TM_CCOEFF': cv2.TM_CCOEFF,
                'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
                'TM_CCORR': cv2.TM_CCORR,
                'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
                'TM_SQDIFF': cv2.TM_SQDIFF,
                'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
            }

            self.metodo = metodos[metodo]

            w, h = template.shape[::-1]

            res = cv2.matchTemplate(img, template, self.metodo)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if self.metodo in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                sup_izq = min_loc
            else:
                sup_izq = max_loc
            
            inf_der = (sup_izq[0] + w, sup_izq[1] + h)

            cv2.rectangle(imgOri, sup_izq, inf_der, (0, 255, 0), 2)

            return imgOri
        except Exception as e:
            msg.error_message(f"Error en Template Matching: {str(e)}")