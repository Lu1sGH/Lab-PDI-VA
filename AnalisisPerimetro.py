import cv2
import numpy as np
import Messages as msg

class AnalisisPerimetro:
    """
    Clase para análisis de perímetro de objetos en imágenes usando OpenCV.
    Proporciona métodos para calcular, visualizar y analizar perímetros de contornos.
    """
    
    def __init__(self):
        pass
    
    def _obtener_contornos_filtrados(self, img_gris):
        """Función auxiliar para obtener contornos filtrados correctamente."""
        h, w = img_gris.shape
        area_imagen = h * w
        
        # Intentar binarización normal
        _, binaria1 = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contornos1, _ = cv2.findContours(binaria1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Intentar binarización invertida
        _, binaria2 = cv2.threshold(img_gris, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        contornos2, _ = cv2.findContours(binaria2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Usar el conjunto de contornos que tenga más elementos válidos
        def filtrar_contornos(contornos):
            filtrados = []
            for contorno in contornos:
                area = cv2.contourArea(contorno)
                # Eliminar contornos muy pequeños (menos del 0.05% del área de la imagen)
                if area < area_imagen * 0.0005:
                    continue
                # Eliminar contornos que ocupen más del 98% de la imagen (probablemente el borde)
                if area > area_imagen * 0.98:
                    continue
                # Eliminar contornos que toquen los bordes de la imagen (probablemente el marco)
                x, y, w_rect, h_rect = cv2.boundingRect(contorno)
                if x <= 2 or y <= 2 or (x + w_rect) >= (w - 2) or (y + h_rect) >= (h - 2):
                    # Verificar si realmente es un objeto en el borde o el marco
                    if area > area_imagen * 0.5:
                        continue
                filtrados.append(contorno)
            return filtrados
        
        filtrados1 = filtrar_contornos(contornos1)
        filtrados2 = filtrar_contornos(contornos2)
        
        # Retornar el conjunto con más contornos válidos
        if len(filtrados2) > len(filtrados1):
            return filtrados2, binaria2
        return filtrados1, binaria1
    
    def analizar_perimetro(self, imagen):
        """
        Analiza el perímetro de todos los objetos en la imagen.
        Retorna una imagen con los contornos dibujados y el perímetro calculado.
        
        Args:
            imagen: Imagen de entrada (BGR o escala de grises)
            
        Returns:
            Imagen con contornos y perímetros visualizados
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(imagen.shape) == 3:
                img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                img_gris = imagen.copy()
            
            # Obtener contornos filtrados
            contornos_filtrados, _ = self._obtener_contornos_filtrados(img_gris)
            
            # Crear imagen resultado en color
            if len(imagen.shape) == 3:
                resultado = imagen.copy()
            else:
                resultado = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
            
            # Si no hay contornos, mostrar mensaje
            if len(contornos_filtrados) == 0:
                msg.alerta_message("No se encontraron objetos en la imagen. Asegúrese de que la imagen tenga figuras visibles.")
                return resultado
            
            # Dibujar contornos y calcular perímetros
            for i, contorno in enumerate(contornos_filtrados):
                # Calcular perímetro
                perimetro = cv2.arcLength(contorno, True)
                
                # Dibujar contorno
                cv2.drawContours(resultado, [contorno], -1, (0, 255, 0), 2)
                
                # Obtener punto para mostrar texto
                M = cv2.moments(contorno)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(resultado, f'Obj{i+1}: P={perimetro:.1f}', (cx-60, cy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            
            return resultado
            
        except Exception as e:
            msg.error_message(f"Error al analizar el perímetro: {str(e)}")
            print(f"Error al analizar el perímetro: {str(e)}")
            return None
    
    def calcular_perimetro_exacto(self, imagen):
        """Calcula el perímetro exacto usando cv2.arcLength con aproximación de contorno cerrado."""
        try:
            # Convertir a escala de grises si es necesario
            if len(imagen.shape) == 3:
                img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                img_gris = imagen.copy()
            
            # Obtener contornos filtrados
            contornos_filtrados, _ = self._obtener_contornos_filtrados(img_gris)
            
            # Crear imagen resultado
            if len(imagen.shape) == 3:
                resultado = imagen.copy()
            else:
                resultado = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
            
            # Si no hay contornos, retornar imagen original
            if len(contornos_filtrados) == 0:
                return resultado
            
            # Calcular y dibujar perímetros
            for i, contorno in enumerate(contornos_filtrados):
                # Calcular perímetro exacto (contorno cerrado)
                perimetro = cv2.arcLength(contorno, True)
                
                # Aproximar contorno para mejor visualización
                epsilon = 0.02 * perimetro
                aproximado = cv2.approxPolyDP(contorno, epsilon, True)
                
                # Dibujar contorno aproximado
                cv2.drawContours(resultado, [aproximado], -1, (255, 0, 0), 2)
                
                # Dibujar puntos del contorno
                for punto in aproximado:
                    cv2.circle(resultado, tuple(punto[0]), 3, (0, 0, 255), -1)
                
                # Mostrar información del perímetro
                M = cv2.moments(contorno)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(resultado, f'Obj{i+1}: P={perimetro:.1f}', (cx-60, cy+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            return resultado
            
        except Exception as e:
            msg.error_message(f"Error al calcular perímetro exacto: {str(e)}")
            print(f"Error al calcular perímetro exacto: {str(e)}")
            return None
    
    def analizar_perimetro_y_area(self, imagen):
        """
        Analiza tanto el perímetro como el área de los objetos.
        Calcula métricas adicionales como compacidad (4π*área/perímetro²).
        """
        try:
            # Convertir a escala de grises si es necesario
            if len(imagen.shape) == 3:
                img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                img_gris = imagen.copy()
            
            # Obtener contornos filtrados
            contornos_filtrados, _ = self._obtener_contornos_filtrados(img_gris)
            
            # Crear imagen resultado
            if len(imagen.shape) == 3:
                resultado = imagen.copy()
            else:
                resultado = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
            
            # Si no hay contornos, retornar imagen original
            if len(contornos_filtrados) == 0:
                return resultado
            
            # Analizar cada contorno
            for i, contorno in enumerate(contornos_filtrados):
                # Calcular perímetro y área
                perimetro = cv2.arcLength(contorno, True)
                area = cv2.contourArea(contorno)
                
                # Calcular compacidad (métrica circular: 1 = círculo perfecto)
                if perimetro > 0:
                    compacidad = (4 * np.pi * area) / (perimetro ** 2)
                else:
                    compacidad = 0
                
                # Obtener rectángulo delimitador
                x, y, w, h = cv2.boundingRect(contorno)
                
                # Dibujar contorno
                cv2.drawContours(resultado, [contorno], -1, (0, 255, 0), 2)
                
                # Dibujar rectángulo delimitador
                cv2.rectangle(resultado, (x, y), (x+w, y+h), (255, 0, 0), 1)
                
                # Mostrar información
                info_text = [
                    f'Obj {i+1}',
                    f'P: {perimetro:.1f}',
                    f'A: {area:.1f}',
                    f'C: {compacidad:.3f}'
                ]
                
                for j, texto in enumerate(info_text):
                    cv2.putText(resultado, texto, (x, y - 60 + j*15), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            
            return resultado
            
        except Exception as e:
            msg.error_message(f"Error al analizar perímetro y área: {str(e)}")
            print(f"Error al analizar perímetro y área: {str(e)}")
            return None
    
    def perimetro_con_aproximacion(self, imagen, epsilon_factor=0.02):
        """Calcula el perímetro usando aproximación poligonal del contorno."""
        try:
            # Convertir a escala de grises si es necesario
            if len(imagen.shape) == 3:
                img_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
            else:
                img_gris = imagen.copy()
            
            # Obtener contornos filtrados
            contornos_filtrados, _ = self._obtener_contornos_filtrados(img_gris)
            
            # Crear imagen resultado
            if len(imagen.shape) == 3:
                resultado = imagen.copy()
            else:
                resultado = cv2.cvtColor(imagen, cv2.COLOR_GRAY2BGR)
            
            # Si no hay contornos, retornar imagen original
            if len(contornos_filtrados) == 0:
                return resultado
            
            # Procesar cada contorno
            for i, contorno in enumerate(contornos_filtrados):
                # Calcular perímetro original
                perimetro_original = cv2.arcLength(contorno, True)
                
                # Aproximar contorno
                epsilon = epsilon_factor * perimetro_original
                contorno_aproximado = cv2.approxPolyDP(contorno, epsilon, True)
                
                # Calcular perímetro aproximado
                perimetro_aproximado = cv2.arcLength(contorno_aproximado, True)
                
                # Dibujar contorno original (verde)
                cv2.drawContours(resultado, [contorno], -1, (0, 255, 0), 1)
                
                # Dibujar contorno aproximado (azul más grueso)
                cv2.drawContours(resultado, [contorno_aproximado], -1, (255, 0, 0), 2)
                
                # Obtener centroide para mostrar información
                M = cv2.moments(contorno)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    info = f'Obj{i+1}: P_orig={perimetro_original:.1f} P_apr={perimetro_aproximado:.1f}'
                    cv2.putText(resultado, info, (cx-100, cy), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            return resultado
            
        except Exception as e:
            msg.error_message(f"Error en aproximación de perímetro: {str(e)}")
            print(f"Error en aproximación de perímetro: {str(e)}")
            return None

