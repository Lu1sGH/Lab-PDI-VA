import Messages as msg
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os

class RecObjs:

    def __init__(self, dataset_path="data.csv", k=3):
        """
        Inicializa y entrena el modelo KNN automáticamente
        
        Args:
            dataset_path: ruta al archivo CSV con el dataset
            k: número de vecinos para KNN (default=3)
        """
        self.knn_model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
        # Entrenar el modelo automáticamente al iniciar
        if os.path.exists(dataset_path):
            self.train_model(dataset_path, k)
        else:
            msg.error_message(f"No se encontró el archivo {dataset_path}")
            print(f"⚠ Advertencia: No se encontró el archivo {dataset_path}")
    
    def train_model(self, dataset_path, k=3):
        """
        Entrena el modelo KNN con el dataset proporcionado
        
        Args:
            dataset_path: ruta al CSV con los datos
            k: número de vecinos para KNN (default=3)
        """
        try:
            # Cargar dataset
            df = pd.read_csv(dataset_path)
            
            # Separar features (h1-h7, d1-d10) y labels (clase)
            X = df.iloc[:, :-1].values  # Todas las columnas excepto la última
            y = df.iloc[:, -1].values    # Última columna (clase)
            
            # Normalizar los datos
            X_scaled = self.scaler.fit_transform(X)
            
            # Entrenar KNN
            self.knn_model = KNeighborsClassifier(n_neighbors=k)
            self.knn_model.fit(X_scaled, y)
            
            self.is_trained = True
            
            msg.todobien_message(f"Modelo KNN entrenado exitosamente con {len(X)} muestras")
            print(f"✓ Modelo KNN entrenado correctamente")
            print(f"  - Muestras de entrenamiento: {len(X)}")
            print(f"  - Vecinos (k): {k}")
            print(f"  - Clases: 0=Círculo, 1=Cuadrado")
            
        except Exception as e:
            msg.error_message(f"Error al entrenar el modelo: {str(e)}")
            print(f"✗ Error al entrenar el modelo: {str(e)}")

    def genVec(self, img, descObj):
        """
        Genera el vector de características y predice la clase
        
        Args:
            img: imagen de entrada
            descObj: objeto descriptor con métodos descFourier y momentosHU
            
        Returns:
            tuple: (vector, clase_predicha, probabilidades, nombre_clase)
                   Si el modelo no está entrenado, retorna (vector, None, None, None)
        """
        try:
            if descObj is None:
                raise ValueError("No se ha proporcionado ningún descriptor de objeto.")
            
            # Calcular descriptores
            vDF = descObj.descFourier(img, precision=10)
            vDF = vDF.astype(np.float64)
            momHu = descObj.momentosHU(img, verboose=False)
            
            # Construir vector de características
            vecgen = []
            
            # Agregar momentos de Hu (h1-h7)
            for i in momHu:
                vecgen.append(float(i[0]))
            
            # Agregar descriptores de Fourier (d1-d10)
            for i in vDF:
                vecgen.append(float(i))
            
            print(f"Vector generado: {vecgen}")
            print(f"Longitud del vector: {len(vecgen)}")
            
            # Si el modelo no está entrenado, solo retornar el vector
            if not self.is_trained:
                msg.warning_message("El modelo KNN no ha sido entrenado. No se puede clasificar.")
                print("⚠ Modelo no entrenado - no se puede realizar predicción")
                return vecgen, None, None, None
            
            # Normalizar el vector con el mismo escalador del entrenamiento
            vec_array = np.array(vecgen).reshape(1, -1)
            vec_scaled = self.scaler.transform(vec_array)
            
            # Realizar predicción
            prediction = self.knn_model.predict(vec_scaled)[0]
            probabilities = self.knn_model.predict_proba(vec_scaled)[0]
            
            # Interpretar resultado
            clase_nombre = "Círculo" if prediction == 0 else "Cuadrado"
            
            print(f"\n{'='*60}")
            print(f"  RESULTADO DE CLASIFICACIÓN")
            print(f"{'='*60}")
            print(f"  Predicción: {clase_nombre} (clase {prediction})")
            print(f"  Confianza:")
            print(f"    - Círculo: {probabilities[0]:.2%}")
            print(f"    - Cuadrado: {probabilities[1]:.2%}")
            print(f"{'='*60}\n")
            
            return vecgen, int(prediction), probabilities, clase_nombre
            
        except Exception as e:
            msg.error_message(f"Error en la función Generar Vectores: {str(e)}")
            print(f"✗ Error en la función Generar Vectores: {str(e)}")
            return None, None, None, None