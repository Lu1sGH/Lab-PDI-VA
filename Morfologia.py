import Messages as msg
import numpy as np
import matplotlib.pyplot as plt
import cv2

class Mofologia:
    def __init__(self):
        # Definir el kernel (EE)
        self.kernel = np.ones((3,3), np.uint8)
        self.kT = [
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
            [0,0,1,0,0,1,1,1,1,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        ]
        self.kT = np.array(self.kT, dtype=np.uint8)
        self.kernel = self.kT.copy()
        self.kernel = np.where(self.kernel == 0, -1, 1)

    def setEE(self, kernelC):
        self.kernel = kernelC

    def erosionCV(self, imagen):
        # Aplicar la operación de erosión
        imagen_erosionada = cv2.erode(imagen, self.kernel, iterations = 1)
        # Mostrar la imagen erosionada
        return imagen_erosionada

    def dilatacionCV(self, imagen):
        # Aplicar la operación de dilatación

        #Definir el centro del kernel
        centro = (self.kernel.shape[1] // 2, self.kernel.shape[0] // 2)

        """if(self.kernel.shape[0] == 7 and self.kernel.shape[1] == 16):
            self.kernel = self.kT.copy()
            #Voltear la figura para compensar la inversión de la dilatación
            self.kernel = cv2.flip(self.kT, -1)"""

        imagen_dilatada = cv2.dilate(imagen, self.kernel, anchor=centro, iterations=1)
        # Mostrar la imagen dilatada
        return imagen_dilatada
    
    def aperturaCV(self, imagen):
        # Aplicar la operación de apertura usando la función definida para ello en Open CV
        imagen_apertura = cv2.morphologyEx(imagen, cv2.MORPH_OPEN, self.kernel)
        # Mostrar la imagen con apertura
        return imagen_apertura

    def cierreCV(self, imagen):
        # Aplicar la operación de cierre
        imagen_cierre = cv2.morphologyEx(imagen, cv2.MORPH_CLOSE, self.kernel)
        # Mostrar la imagen con cierre
        return imagen_cierre
    
    def hitOrMissCV(self, imagen): #Aplicar la operación de hit or miss
        if(len(imagen.shape) == 3):
            msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        # Aplicar la operación de hit or miss
        imagen_hit_or_miss = cv2.morphologyEx(imagen, cv2.MORPH_HITMISS, self.kernel)
        # Mostrar la imagen con hit or miss
        return imagen_hit_or_miss
    
    def fronteraInteriorCV(self, imagen): #Calcular la frontera interior
        dilatada = self.dilatacionCV(imagen)
        frontera = cv2.bitwise_xor(imagen, dilatada)
        return frontera
    
    def fronteraExteriorCV(self, imagen): #Calcular la frontera exterior
        erosionada = self.erosionCV(imagen)
        frontera = cv2.bitwise_xor(erosionada, imagen)
        return frontera
    
    def apertura(self, imagen): #Calcular apertura sin el uso de la función de Open CV
        if(len(imagen.shape) == 3):
            msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        erosionada = self.erosionCV(imagen)
        dilatada = self.dilatacionCV(erosionada)
        return dilatada

    def cierre(self, imagen): #Calcular cierre sin el uso de la función de Open CV
        if(len(imagen.shape) == 3):
            msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        dilatada = self.dilatacionCV(imagen)
        erosionada = self.erosionCV(dilatada)
        return erosionada

    def adelgazamiento(self, image): #Adelgazamiento de una imagen con openCV
        if len(image.shape) == 3:
            msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        thin = cv2.ximgproc.thinning(image)
        return thin

    def esqueleto(self, imagen): #Calcular el esqueleto de una imagen con openCV
        if len(imagen.shape) == 3:
            msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        sk = cv2.ximgproc.thinning(imagen, thinningType=cv2.ximgproc.THINNING_GUOHALL)

        return sk
    
    def tophat(self, imagen): #Calcular la operación tophat de una imagen con openCV
        if len(imagen.shape) == 3:
            msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        tophat = cv2.subtract(imagen, self.aperturaCV(imagen))
        return tophat

    def blackhat(self, imagen): #Calcular la operación blackhat de una imagen con openCV
        if len(imagen.shape) == 3:
            msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        blackhat = cv2.subtract(self.cierreCV(imagen), imagen)
        return blackhat
    
    def gradSim(self, imagen): #Calcular el gradiente simétrico de una imagen con openCV
        if len(imagen.shape) == 3:
            msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        dilatada = self.dilatacionCV(imagen)
        erosionada = self.erosionCV(imagen)
        gradiente = cv2.subtract(dilatada, erosionada)
        return gradiente
"""
    def esqueleto(self, img):
        if len(img.shape) == 3:
            msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        tam = np.size(img)
        skel = np.zeros(img.shape,np.uint8)
        
        _, img = cv2.threshold(img,127,255,0)
        element = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
        done = False
        
        while not done:
            erosionada = cv2.erode(img,element)
            temp = cv2.dilate(erosionada,element)
            temp = cv2.subtract(img,temp)
            skel = cv2.bitwise_or(skel,temp)
            img = erosionada.copy()
        
            zeros = tam - cv2.countNonZero(img)
            if zeros==tam:
                done = True

        return skel

    def adelgazamiento(self, imagen):
        if len(imagen.shape) == 3:
            msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
        thin = np.zeros(imagen.shape,dtype='uint8')
        

        img1 = imagen.copy()
        while (cv2.countNonZero(img1)!=0):
            er = cv2.erode(img1,kernel)
            aper = cv2.morphologyEx(er,cv2.MORPH_OPEN,kernel)
            sub = er - aper
            thin = cv2.bitwise_or(sub,thin)
            img1 = er.copy()

        return thin
    
    def thinUseless(self, imagen):
        if len(imagen.shape) == 3:
            msg.alerta_message("El método no admite imágenes a color. La imagen se convertirá a grises para su uso.")
            imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

        thin = cv2.bitwise_and(imagen, cv2.bitwise_not(self.hitOrMissCV(imagen)))

        return thin
"""