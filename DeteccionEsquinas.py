"""
Módulo para detección de esquinas usando el algoritmo de Harris
"""
import cv2
import numpy as np
from matplotlib import pyplot as plt
from scipy import signal
from scipy.ndimage import maximum_filter

class DeteccionEsquinas:
    def __init__(self):
        self.blockSize = 3
        self.ksize = 5
        self.k = 0.04
    
    def setParametros(self, blockSize, ksize, k):
        """Establece los parámetros para el detector Harris"""
        self.blockSize = blockSize
        self.ksize = ksize
        self.k = k
    
    def harris_manual(self, img):
        """
        Implementación manual mejorada del detector de esquinas Harris.
        Se busca aproximar al máximo a cv2.cornerHarris.
        """

        # 1) Convertir a gris
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_color = img.copy()
        else:
            gray = img.copy()
            img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        # 2) Preprocesamiento (copiando lógica típica de OpenCV)
        gray = cv2.GaussianBlur(gray, (3,3), 0)
        gray_f = np.float32(gray)

        # 3) Gradientes con Sobel 
        Ix = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=self.ksize)
        Iy = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=self.ksize)

        # 4) Productos de derivadas
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy

        # 5) Ventana gaussiana
        sigma = 0.3*((self.blockSize-1)*0.5 - 1) + 0.8
        window = cv2.getGaussianKernel(self.blockSize, sigma)
        window = window @ window.T

        # 6) Suavizado
        Sxx = cv2.filter2D(Ixx, -1, window)
        Syy = cv2.filter2D(Iyy, -1, window)
        Sxy = cv2.filter2D(Ixy, -1, window)

        # 7) Respuesta Harris real:
        #    R = det(M) - k * (trace(M))^2
        detM = Sxx * Syy - Sxy * Sxy
        traceM = Sxx + Syy
        R = detM - self.k * (traceM * traceM)

        # 8) SUPRESIÓN DE NO-MÁXIMOS
        R = cv2.dilate(R, None)  # expansión leve
        R_norm = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX)
        R_norm = np.uint8(R_norm)

        # 9) Umbral 
        umbral = 0.01 * R.max()
        _, R_bin = cv2.threshold(R, umbral, 255, cv2.THRESH_BINARY)
        R_bin = np.uint8(R_bin)

        # 10) Componentes conectados
        num, labels, stats, centroids = cv2.connectedComponentsWithStats(R_bin)

        # 11) Filtrar por tamaño
        MIN_AREA = 10
        MAX_AREA = 2000
        centros_validos = []
        for i in range(1, num):
            if MIN_AREA < stats[i, cv2.CC_STAT_AREA] < MAX_AREA:
                centros_validos.append(centroids[i])
        centros_validos = np.array(centros_validos, dtype=np.float32)

        if len(centros_validos) == 0:
            print("No se detectaron esquinas")
            return img_color

        # 12) REFINAMIENTO SUBPIXEL
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            50,
            0.01
        )

        esquinas_refinadas = cv2.cornerSubPix(
            gray_f,
            centros_validos,
            winSize=(5,5),
            zeroZone=(-1,-1),
            criteria=criteria
        )

        # 13) Dibujar resultados
        img_res = img_color.copy()
        for (cx, cy), (rx, ry) in zip(centros_validos, esquinas_refinadas):
            cv2.circle(img_res, (int(rx), int(ry)), 3, (0,255,0), -1)

        print(np.hstack((centros_validos[:10], esquinas_refinadas[:10])))

        return img_res
    
    def harris_opencv(self, img):
        """
        Detector de esquinas Harris usando cv2.cornerHarris
        """
        # Convertir a escala de grises (por si acaso)
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = img.copy()
        
        img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img.copy()
        
        # Preprocesamiento
        img_eq = cv2.equalizeHist(img_gray)
        img_denoised = cv2.bilateralFilter(img_eq, 9, 75, 75)
        img_float = np.float32(img_denoised)
        
        # Detectar esquinas con Harris
        esquinas = cv2.cornerHarris(img_float, self.blockSize, self.ksize, self.k)
        
        # Normalizar para visualización
        esquinas_norm = cv2.normalize(esquinas, None, 0, 255, cv2.NORM_MINMAX)
        
        # Dilatar
        kernel = np.ones((3,3), np.uint8)
        esquinas = cv2.dilate(esquinas, kernel, iterations=2)
        
        # Umbral adaptativo
        umbral = np.percentile(esquinas[esquinas > 0], 99)
        _, esquinas_bin = cv2.threshold(esquinas, umbral, 255, cv2.THRESH_BINARY)
        esquinas_bin = np.uint8(esquinas_bin)
        
        # Componentes conectados
        componentes, labels, stats, centroids = cv2.connectedComponentsWithStats(esquinas_bin)
        
        # Filtrar por área
        MIN_AREA = 5
        MAX_AREA = 500
        esquinas_filtradas = []
        
        for i in range(1, componentes):
            area = stats[i, cv2.CC_STAT_AREA]
            if MIN_AREA < area < MAX_AREA:
                esquinas_filtradas.append(centroids[i])
        
        esquinas_filtradas = np.array(esquinas_filtradas)
        
        if len(esquinas_filtradas) > 0:
            # Refinar con subpíxel
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)
            corners = cv2.cornerSubPix(img_float, np.float32(esquinas_filtradas), 
                                       winSize=(5,5), zeroZone=(-1,-1), criteria=criteria)
            
            res = np.hstack((esquinas_filtradas, corners))
            res_int = res.astype(int)
            
            # Dibujar resultado
            img_resultado = img_color.copy()
            
            for i in range(len(res_int)):
                # Esquina refinada en verde
                cv2.circle(img_resultado, (res_int[i,2], res_int[i,3]), 5, (0,255,0), -1)
            
            return img_resultado
        else:
            print("No se detectaron esquinas. Intenta ajustar los parámetros.")
            return img_color