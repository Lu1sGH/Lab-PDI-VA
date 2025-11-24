import pandas as pd
import os

# Ruta relativa del archivo CSV
ruta_csv = "data.csv"

# Ruta absoluta del archivo CSV
ruta_absoluta = os.path.abspath(ruta_csv)

# Directorio donde se encuentra el archivo
directorio = os.path.dirname(ruta_absoluta)
print("="*70)
print("CARGA DE ARCHIVO CSV CON PANDAS")
print("="*70)

# Verificar si el archivo existe
if os.path.exists(ruta_csv):
    print(f"\nArchivo encontrado")
    print(f"\nRuta relativa: {ruta_csv}")
    print(f"Ruta absoluta: {ruta_absoluta}")
    print(f"Directorio: {directorio}")
    
    # Cargar el CSV con pandas
    df = pd.read_csv(ruta_csv)
    
else:
    print(f"Error: No se encontró el archivo en la ruta: {os.path.abspath(ruta_csv)}")
    print("Por favor, verifica que el archivo data.csv esté en el directorio del proyecto.")

