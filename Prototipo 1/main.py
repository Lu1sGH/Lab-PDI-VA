# paquetes para descargar
# pip install numpy matplotlib opencv-python

from ecualizacion import Ecualizador
from operaciones import Operaciones

def main():
    ruta_imagen_principal = 'img2.jpeg'

    try:
        # Parte 1: Ecualización
        eq = Ecualizador(ruta_imagen_principal)
        eq.mostrar_original()
        eq.mostrar_histograma_original()
        eq.ecualizar_uniformemente()
        eq.mostrar_ecualizada()
        eq.mostrar_histograma_ecualizada()

        # Parte 2: Operaciones Aritméticas y Lógicas
        op = Operaciones(ruta_imagen_principal)
        op.operaciones_aritmeticas()
        op.operaciones_logicas('img1.jpg')
    except Exception as e:
        print(e)

if __name__ == "__main__":
    main()