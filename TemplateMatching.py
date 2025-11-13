import Messages as msg
import numpy as np
import cv2

class TemplateMatching:
    def __init__(self):
        self.metodos = {
            'TM_CCOEFF': cv2.TM_CCOEFF,
            'TM_CCOEFF_NORMED': cv2.TM_CCOEFF_NORMED,
            'TM_CCORR': cv2.TM_CCORR,
            'TM_CCORR_NORMED': cv2.TM_CCORR_NORMED,
            'TM_SQDIFF': cv2.TM_SQDIFF,
            'TM_SQDIFF_NORMED': cv2.TM_SQDIFF_NORMED
        }

    def tm_OpenCV(self, img, template, metodo=''):
        try:
            imgOri = img.copy()
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            selMet = self.metodos[metodo]

            w, h = template.shape[::-1]

            res = cv2.matchTemplate(img, template, selMet)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            if selMet in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                sup_izq = min_loc
            else:
                sup_izq = max_loc
            
            inf_der = (sup_izq[0] + w, sup_izq[1] + h)

            cv2.rectangle(imgOri, sup_izq, inf_der, (0, 255, 0), 2)

            return imgOri
        except Exception as e:
            msg.error_message(f"Error en la función Template Matching con OpenCV: {str(e)}")

    def tm_Manual(self, img, template, metodo=''):
        try:
            imgOri = img.copy()
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if len(template.shape) == 3:
                template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            w, h = template.shape[::-1]
            res = np.zeros((img.shape[0] - h + 1, img.shape[1] - w + 1), dtype=np.float32)
            img = img.astype(np.float32)
            template = template.astype(np.float32)

            esquina = (0, 0, np.inf) if metodo in ['TM_SQDIFF', 'TM_SQDIFF_NORMED'] else (0, 0, -np.inf)

            if metodo == 'TM_SQDIFF':
                for x in range(img.shape[0] - h + 1):
                    for y in range(img.shape[1] - w + 1):
                        I = img[x:x+h, y:y+w]
                        rsq_diff = np.sum((template - I)**2)
                        res[x, y] = rsq_diff
                        if esquina[2] > rsq_diff:
                            esquina = (x, y, rsq_diff)

            elif metodo == 'TM_SQDIFF_NORMED':
                for x in range(img.shape[0] - h + 1):
                    for y in range(img.shape[1] - w + 1):
                        I = img[x:x+h, y:y+w]
                        rsq_diff_normed = np.sum((template - I)**2) / np.sqrt(np.sum(template**2) * np.sum(I**2))
                        res[x, y] = rsq_diff_normed
                        if esquina[2] > rsq_diff_normed:
                            esquina = (x, y, rsq_diff_normed)

            elif metodo == 'TM_CCORR':
                for x in range(img.shape[0] - h + 1):
                    for y in range(img.shape[1] - w + 1):
                        I = img[x:x+h, y:y+w]
                        rccorr = np.sum(template*I)
                        res[x, y] = rccorr
                        if esquina[2] < rccorr:
                            esquina = (x, y, rccorr)

            elif metodo == 'TM_CCORR_NORMED':
                for x in range(img.shape[0] - h + 1):
                    for y in range(img.shape[1] - w + 1):
                        I = img[x:x+h, y:y+w]
                        rccorr_normed = np.sum(template*I) / np.sqrt(np.sum(template**2) * np.sum(I**2))
                        res[x, y] = rccorr_normed
                        if esquina[2] < rccorr_normed:
                            esquina = (x, y, rccorr_normed)

            elif metodo == 'TM_CCOEFF':
                for x in range(img.shape[0] - h + 1):
                    for y in range(img.shape[1] - w + 1):
                        I = img[x:x+h, y:y+w]
                        Tprim = template - np.sum(template)/(w*h)
                        Iprim = I - np.sum(I)/(w*h)
                        rccoeff = np.sum(Tprim*Iprim)
                        res[x, y] = rccoeff
                        if esquina[2] < rccoeff:
                            esquina = (x, y, rccoeff)

            elif metodo == 'TM_CCOEFF_NORMED':
                for x in range(img.shape[0] - h + 1):
                    for y in range(img.shape[1] - w + 1):
                        I = img[x:x+h, y:y+w]
                        Tprim = template - np.sum(template)/(w*h)
                        Iprim = I - np.sum(I)/(w*h)
                        rccoeff_normed = np.sum(Tprim*Iprim)/np.sqrt(np.sum(Tprim**2)*np.sum(Iprim**2))
                        res[x, y] = rccoeff_normed
                        if esquina[2] < rccoeff_normed:
                            esquina = (x, y, rccoeff_normed)
            
            inf_der = (esquina[1] + w, esquina[0] + h)

            cv2.rectangle(imgOri, (esquina[1], esquina[0]), inf_der, (0, 255, 0), 2)

            msg.todobien_message(f"Template Matching aplicado correctamente.")
            return imgOri
        except Exception as e:
            msg.error_message(f"Error en la función Template Matching Manual: {str(e)}")