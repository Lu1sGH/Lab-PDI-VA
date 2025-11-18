"""
Módulo para detección y conteo de monedas
"""
import cv2
import numpy as np

class DeteccionMonedas:
    def __init__(self):
        # Rangos de áreas para cada denominación (ajustables según escala)
        self.rangos = {
            "10":  (8000, 9000),   
            "5":   (6500, 7200),
            "2":   (5000, 6000),
            "1":   (4300, 4800),
            "50c": (2900, 4000)
        }
        self.scale = 0.4
        self.min_area = 500
        self.canny_low = 100
        self.canny_high = 300
        self.kernel_size = 4
    
    def setParametros(self, scale, min_area, canny_low, canny_high, kernel_size):
        """Establece los parámetros para la detección"""
        self.scale = scale
        self.min_area = min_area
        self.canny_low = canny_low
        self.canny_high = canny_high
        self.kernel_size = kernel_size
    
    def setRangos(self, rangos):
        """Actualiza los rangos de denominaciones"""
        self.rangos = rangos
    
    def obtener_denominacion(self, area):
        """Determina la denominación de una moneda según su área"""
        for denom, (a_min, a_max) in self.rangos.items():
            if a_min <= area <= a_max:
                return denom
        return "?"
    
    def detectar_monedas(self, img):
        """
        Detecta y cuenta monedas en una imagen
        Retorna: imagen con resultados dibujados, total de dinero, número de monedas
        """
        # Convertir a escala de grises si es necesario
        if len(img.shape) == 3:
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_color = img.copy()
        else:
            img_gray = img.copy()
            img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
        
        # Redimensionar
        img_gray = cv2.resize(img_gray, None, fx=self.scale, fy=self.scale)
        img_color = cv2.resize(img_color, None, fx=self.scale, fy=self.scale)
        
        print("Detectando bordes con Canny...")
        
        # Canny + Closing
        borders = cv2.Canny(img_gray, self.canny_low, self.canny_high)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        closing = cv2.morphologyEx(borders, cv2.MORPH_CLOSE, kernel)
        
        print("Filtrando componentes por área...")
        
        # Componentes conectados
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closing)
        clean = np.zeros_like(closing)
        
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                clean[labels == i] = 255
        
        # Encontrar contornos
        contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Rellenar contornos
        cv2.drawContours(clean, contours, -1, 255, -1)
        
        dinero = 0
        monedas_detectadas = []
        
        print(f"Analizando {len(contours)} monedas...")
        
        # Dibujar resultados
        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            denom = self.obtener_denominacion(area)
            
            if denom != "?":
                if denom == "50c":
                    dinero += 0.5
                else:
                    dinero += int(denom)
            
            monedas_detectadas.append((area, denom))
            
            # Dibujar contorno
            cv2.drawContours(img_color, [cnt], -1, (0, 255, 0), 2)
            
            # Centroide para colocar texto
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = cnt[0][0]
            
            # Escribir el área y la denominación
            texto = f"${denom}"
            cv2.putText(img_color, texto, (cx - 20, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Área en texto 
            texto_area = f"{int(area)}"
            cv2.putText(img_color, texto_area, (cx - 20, cy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
            
            print(f"Moneda {i+1}: área = {int(area)}, denominación = ${denom}")
        
        print(f"\n✓ Total monedas detectadas: {len(contours)}")
        print(f"✓ Total de dinero: ${dinero}")
        
        # Agregar texto resumen en la imagen
        texto_resumen = f"Monedas: {len(contours)} | Total: ${dinero}"
        cv2.putText(img_color, texto_resumen, (90, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        return img_color, dinero, len(contours), monedas_detectadas