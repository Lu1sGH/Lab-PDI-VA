"""
Programa desarrollado por:
Gonzalo 🕊🕊
Luis Z 
Daniel Diaz
Danielle Sophia
"""

#Librearias para la interfaz grafica
import customtkinter as cusTK
from customtkinter import CTkImage
import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog
from PIL import Image, ImageTk
import Messages as msg #Libreria (de creación propia) para mensajes de error y alerta
import PilaCambios as Cambios #Librería (de creación propia) para deshacer cambios a la imagen resultado
import cv2

#Librerias para PDI
from ecualizacion import Ecualizador
from operaciones import Operaciones
from Ruido import Ruido
from Filtros_PB_NL import Filtros_PasoBajas_NoLineales
from Segmentacion import Segmentacion
from Filtros_PA import Filtros_Paso_Altas
from Conteo import Conteo
from Morfologia import Mofologia
from Editor_Kernel import EditorKernel
from Mascaras_Operadores import Mascaras_Operadores
from TemplateMatching import TemplateMatching
from DeteccionEsquinas import DeteccionEsquinas
from AnalisisPerimetro import AnalisisPerimetro
from DeteccionMonedas import DeteccionMonedas

cusTK.set_appearance_mode("Dark")  #Configuración inicial de la apariencia
cusTK.set_default_color_theme("blue")
fuente_global = ("Segoe UI", 13, "bold") #Fuente para todos los botones de la aplicación

class App(cusTK.CTk):
    def __init__(self):
        super().__init__()
        
        #Inicialización de la ventana principal
        self.title("Laboratorio de Imágenes")
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}")
        self.minsize(700, 600)
        self.resizable(True, True)
        
        #Inicialización de variables para la manipulación de imágenes
        self.op = Operaciones() #Instancia de la clase de operaciones
        self.ec = Ecualizador() #Instancia de la clase de ecualización
        self.ruido = Ruido() #Instancia de la clase para añadir ruido
        self.fPBNL = Filtros_PasoBajas_NoLineales() #Instancia de la clase de filtros paso bajas y no lineales
        self.fPA = Filtros_Paso_Altas() #Instancia de la clase de filtros paso altas
        self.seg = Segmentacion() #Instancia de la clase de segmentación
        self.conteo = Conteo() #Instancia de la clase de conteo de objetos
        self.morfologia = Mofologia() #Instancia de la clase de morfología
        self.maskOp = Mascaras_Operadores() #Instancia de la clase de máscaras y operadores
        self.tmO = TemplateMatching() #Instancia de la clase de Template Matching
        self.detEsq = DeteccionEsquinas()  # Instancia de detección de esquinas
        self.harris_blockSize = 3
        self.harris_ksize = 5
        self.harris_k = 0.04
        self.analisisPerimetro = AnalisisPerimetro() #Instancia de la clase de análisis de perímetro
        self.imagen1 = None
        self.imagen2 = None
        self.resultado = None
        self.cambios = Cambios.PilaCambios() #Instancia de la clase para deshacer cambios
        self.imagen_actual = 1  #Elige cual es la imagen que se va a operar. Por defecto, operar con la imagen 1
        self.t_kernel = 3 #Tamaño kernel
        self.c = 0 #C para umbralización adaptativa (mean-C)
        self.const = 0 #Constante para operaciones aritmeticas. También para fil promedio pesado y corrección gamma.
        self.maxSeg = 0 #Constante para número máximo de segmentos en umbralizado por segmentación.
        self.sigma = 0.75 #Constante para filtros Gaussianos.
        self.kirsch_dir = "T" #Dirección para el operador de Kirsch. Por defecto, todas las direcciones.
        self.detMon = DeteccionMonedas()  # Instancia de detección de monedas
        self.monedas_scale = 0.4
        self.monedas_min_area = 500
        self.monedas_canny_low = 100
        self.monedas_canny_high = 300
        self.monedas_kernel_size = 4

        #Barra superior para operaciones de PDI
        self.top_bar = cusTK.CTkFrame(self, height=50)
        self.top_bar.pack(side="top", fill="x")
        
        #Barra superior para operaciones de Visión Artificial
        self.top_bar2 = cusTK.CTkFrame(self, height=50)
        self.top_bar2.pack(side="top", fill="x")

        #Parte principal (contenedor de imágenes y resultados)
        self.content_frame = cusTK.CTkFrame(self)
        self.content_frame.pack(fill="both", expand=True)

        #Frame principal izquierdo
        self.frame_imagen = cusTK.CTkFrame(self.content_frame)
        self.frame_imagen.pack(side="left", fill="both", expand=True, padx=5, pady=5)

        #Subframe para imagen 1
        self.frame_imagen1 = cusTK.CTkFrame(self.frame_imagen)
        self.frame_imagen1.pack(fill="both", expand=True, padx=10, pady=5)

        self.image_label1 = cusTK.CTkLabel(self.frame_imagen1, text="")
        self.image_label1.pack(fill="both", expand=True)

        #Subframe para imagen 2
        self.frame_imagen2 = cusTK.CTkFrame(self.frame_imagen)
        self.frame_imagen2.pack(fill="both", expand=True, padx=10, pady=5)

        self.image_label2 = cusTK.CTkLabel(self.frame_imagen2, text="")
        self.image_label2.pack(fill="both", expand=True)

        #Frame para resultado de las operaciones
        self.frameResultado = cusTK.CTkFrame(self.content_frame)
        self.frameResultado.pack(side="right", fill="both", expand=True, padx=5, pady=5)

        self.resultadoLabel = cusTK.CTkLabel(self.frameResultado, text="")
        self.resultadoLabel.pack(fill="both", expand=True, padx=10, pady=10)

        self.init_componentes_PDI() #Inicializa los componentes de la barra superior de PDI
        self.init_componentes_VA() #Inicializa los componentes de la barra superior de Visión Artificial

        """#Botón para cambiar entre modo claro y oscuro (DEPRECATED)
        #self.toggle_button = cusTK.CTkButton(self.top_bar, text="☀ Modo claro", command=self.toggle_theme, font=fuente_global, hover_color="#171717")
        #self.toggle_button.pack(side="right", padx=10, pady=10)"""

    def init_componentes_PDI(self): #Inicializa los componentes de la barra superior de PDI
        #Menu para archivos
        self.archivos_menu = cusTK.CTkOptionMenu(
            self.top_bar,
            values=["Abrir Imagen", "Guardar Imagen Activa", "Cerrar Imagen Activa"],
            command=self.archivos_action,
            font=fuente_global,
            dropdown_font=fuente_global
        )
        self.archivos_menu.set("📁 Archivos")
        self.archivos_menu.pack(side="left", padx=10, pady=10)

        #Menu para seleccionar imagen. Este menú permite elegir la imagen activa para operar.
        self.selector_menu = cusTK.CTkOptionMenu(
            self.top_bar,
            values=["Imagen 1", "Imagen 2", "Imagen 3 (Resultado)"],
            command=self.cambiar_imagen_actual,
            font=fuente_global,
            dropdown_font=fuente_global
        )
        self.selector_menu.set("💻 Elegir imagen activa")
        self.selector_menu.pack(side="left", padx=10, pady=10)

        #Menu de color. Muestra opciones sobre el color de la imagen activa.
        self.colorObjetos_menu = cusTK.CTkOptionMenu(
            self.top_bar,
            values=["Canales RGB", "Convertir a escala de grises", "Histograma Imagen Activa", "Umbralizar manualmente",
                    "Umbralizar adaptativamente \npor propiedades locales", "Umbralizar adaptativamente \npor partición",
                    "Umbralizar por media", "Umbralizar por Otsu", "Umbralizar por Multiumbralización", "Umbralización por Kapur",
                    "Umbralización banda", "Umbralización por mínimo del histograma",
                    "Contar Objetos", "Segmentación Watershed", "Segmentación K-means", 
                    "Segmentación Mean Shift", "Segmentación GrabCut"],
            command=self.color_action,
            font=fuente_global,
            dropdown_font=fuente_global
        )
        self.colorObjetos_menu.set("🖼 Colores y objetos")
        self.colorObjetos_menu.pack(side="left", padx=10, pady=10)

        #Menu de operaciones
        self.operaciones_menu = cusTK.CTkOptionMenu(
            self.top_bar,
            values=["Suma", "Resta", "Multiplicación", "AND", "OR", "XOR", "NOT", 
                    "Ecualizar Uniformemente", "Ecualización Rayleigh", "Ecualización hipercúbica", 
                    "Ecualización exponencial", "Ecualización logaritmo hiperbólica", "Expansión", "Contracción", 
                    "Corrección Gamma", "Ecualización Adaptativa"],
            command=self.operaciones_action,
            font=fuente_global,
            dropdown_font=fuente_global
        )
        self.operaciones_menu.set("📊 Operaciones")
        self.operaciones_menu.pack(side="left", padx=10, pady=10)

        #Menu para filtros y ruido
        self.filtros_menu = cusTK.CTkOptionMenu(
            self.top_bar,
            values=["Añadir ruido impulsivo", "Añadir ruido Gaussiano", "Añadir ruido multiplicativo", 
                    "Filtro Máximo", "Filtro Mínimo", "Filtro promediador", "Filtro promediador pesado", "Filtro mediana", 
                    "Filtro bilateral", "Filtro Gaussiano", "Filtro de Canny"],
            command=self.filtros_action,
            font=fuente_global,
            dropdown_font=fuente_global
        )
        self.filtros_menu.set("🎇 Filtros y ruido")
        self.filtros_menu.pack(side="left", padx=10, pady=10)

        #Menu para Mascaras y Operadores
        self.mas_op_menu = cusTK.CTkOptionMenu(
            self.top_bar,
            values=["Operador de Sobel", "Operador de Prewitt", "Operador de Roberts", "Operador de Laplace",
                    "Mascaras de Kirsch", "Máscaras de Robinson", "Máscaras de Frei-Chen"],
            command=self.mascaras_y_operadores_action,
            font=fuente_global,
            dropdown_font=fuente_global
        )
        self.mas_op_menu.set("🎭 Máscaras y Operadores")
        self.mas_op_menu.pack(side="left", padx=10, pady=10)

        #Menu para Morfología
        self.morfologia_menu = cusTK.CTkOptionMenu(
            self.top_bar,
            values=["Erosión", "Dilatación", "Apertura", "Cierre", "Hit or Miss", "Frontera Interior", "Frontera Exterior",
                    "Apertura tradicional", "Cierre tradicional", "Thinning", "Esqueleto", "Tophat", "Blackhat", "Gradiente simétrico"],
            command=self.morfologia_action,
            font=fuente_global,
            dropdown_font=fuente_global
        )
        self.morfologia_menu.set("🔳 Morfología")
        self.morfologia_menu.pack(side="left", padx=10, pady=10)

        #Botón para elemento estructural
        self.elemEst = cusTK.CTkButton(self.top_bar, text="EE", command=self.elementoEstructural_action, font=fuente_global, width=20)
        self.elemEst.pack(side="left", padx=10, pady=10)

        #Botón para ajustar constantes
        self.cons_boton = cusTK.CTkButton(self.top_bar, text="⚙ Constantes", command=self.setConstantes, font=fuente_global, hover_color="#0A380A")
        self.cons_boton.pack(side="left", padx=10, pady=10)

        #Boton para deshacer cambios
        self.deshacer_boton = cusTK.CTkButton(self.top_bar, text="↺", command=self.deshacerCambios, font=fuente_global, width=30, hover_color="#851717")
        self.deshacer_boton.pack(side="left", padx=10, pady=10)

    def init_componentes_VA(self): #Inicializa los componentes de la barra superior de Visión Artificial
        #Menu para template matching
        self.tm_menu = cusTK.CTkOptionMenu(
            self.top_bar2,
            values=["Regular", "Normalizada", "Correlación", 
                    "Correlación\n Normalizada", "Coeficientes de\n Correlación", 
                    "Coeficientes de\n Correlación\n Normalizada",
                    "R Manual", "RN Manual", "C Manual",
                    "CN Manual", "CC Manual", "CCN Manual"],
            command=self.tm_action,
            font=fuente_global,
            dropdown_font=fuente_global
        )
        self.tm_menu.set("🥅 Temp Match")
        self.tm_menu.pack(side="left", padx=10, pady=10)
        
        # Menu para detección de esquinas
        self.deteccion_menu = cusTK.CTkOptionMenu(
            self.top_bar2,
            values=["Harris (OpenCV)", "Harris Manual", "Detectar Monedas"],
            command=self.detection_action,
            font=fuente_global,
            dropdown_font=fuente_global
        )
        self.deteccion_menu.set("🔍 Detección")
        self.deteccion_menu.pack(side="left", padx=10, pady=10)

        #Menu para análisis de perímetro
        self.perimetro_menu = cusTK.CTkOptionMenu(
            self.top_bar2,
            values=["Analizar Perímetro", "Perímetro Exacto", "Perímetro y Área", 
                    "Perímetro con Aproximación"],
            command=self.perimetro_action,
            font=fuente_global,
            dropdown_font=fuente_global
        )
        self.perimetro_menu.set("📐 Análisis Perímetro")
        self.perimetro_menu.pack(side="left", padx=10, pady=10)

    """ DEPRECATED
    def toggle_theme(self): #Cambiar entre modo claro y oscuro
        try:
            if cusTK.get_appearance_mode() == "Light":
                cusTK.set_appearance_mode("Dark")
                self.toggle_button.configure(text="☀ Modo claro")# Modo guerra
            else:
                cusTK.set_appearance_mode("Light")
                self.toggle_button.configure(text="🌙 Modo oscuro")# 🗣🗣🗣
        except Exception as e:
            msg.error_message(f"Error al cambiar el tema: {str(e)}")
            print(f"Error al cambiar el tema: {str(e)}")"""

    def deshacerCambios(self):
        des = self.cambios.deshacer()
        self.setResultado(des, esDesCambio = True)

    def cambiar_imagen_actual(self, seleccion):
        if seleccion == "Imagen 1":
            self.imagen_actual = 1
        elif seleccion == "Imagen 2":
            self.imagen_actual = 2
        elif seleccion == "Imagen 3 (Resultado)":
            self.imagen_actual = 3

        #Reset de los menús
        self.archivos_menu.set("📁 Archivos")
        self.colorObjetos_menu.set("🖼 Colores y objetos")
        self.operaciones_menu.set("📊 Operaciones")
        self.filtros_menu.set("🎇 Filtros y ruido")
        self.mas_op_menu.set("🎭 Máscaras y Operadores")
        self.morfologia_menu.set("🔳 Morfología")
        self.tm_menu.set("🥅 Temp Match")
        self.deteccion_menu.set("🔍 Detección")
        self.perimetro_menu.set("📐 Análisis Perímetro")

    def obtener_imagen_actual(self):
        try:
            if self.imagen_actual == 1:
                return self.imagen1
            elif self.imagen_actual == 2:
                return self.imagen2
            elif self.imagen_actual == 3:
                return self.resultado
        except Exception as e:
            msg.error_message(f"Error al obtener la imagen actual: {str(e)}")
            print(f"Error al obtener la imagen actual: {str(e)}")
            return None

    def archivos_action(self, choice): #Acciones del menú de archivos
        if choice == "Abrir Imagen":
            self.abrir_imagen()
        elif choice == "Guardar Imagen Activa":
            self.guardar_imagen()
        elif choice == "Cerrar Imagen Activa":
            self.cerrar_imagen()

    def color_action(self, choice): #Acciones del menú de color
        try:
            actual = self.obtener_imagen_actual()
            if actual is None: 
                msg.alerta_message("No se ha cargado una imagen.")
                return
            
            if self.resultado is not None: #Si hay un resultado, se guarda en la pila de cambios para que no se pierda
                self.cambios.guardar(self.resultado.copy())
            
            if choice == "Canales RGB":
                resultado = self.op.mostrar_componentes_RGB(imagen=actual)
            elif choice == "Convertir a escala de grises":
                resultado = self.op.aGris(imagen=actual)
                self.setResultado(resultado)
            elif choice == "Histograma Imagen Activa":
                self.op.mostrar_histograma(actual)
            elif choice == "Umbralizar manualmente":
                self.elegir_umbral()
            elif choice == "Umbralizar adaptativamente \npor propiedades locales":
                resultado = self.seg.umbralizacionAdaptativa(actual, kernel = self.t_kernel , c = self.c)
                self.setResultado(resultado)
            elif choice == "Umbralizar adaptativamente \npor partición":
                resultado = self.seg.umbraladoSegmentacion(actual, self.maxSeg)
                self.setResultado(resultado)
            elif choice == "Umbralizar por media":
                resultado = self.seg.segmentacionUmbralMedia(actual)
                self.setResultado(resultado)
            elif choice == "Umbralizar por Otsu":
                resultado = self.seg.segmentacionOtsu(actual)
                self.setResultado(resultado)
            elif choice == "Umbralizar por Multiumbralización":
                resultado = self.seg.segmentacionMultiumbral(actual, self.maxSeg)
                self.setResultado(resultado)
            elif choice == "Umbralización por Kapur":
                resultado = self.seg.segmentacionKapur(actual)
                self.setResultado(resultado)
            elif choice == "Umbralización banda":
                resultado = self.seg.segmentacionUmbralBanda(actual)
                self.setResultado(resultado)
            elif choice == "Umbralización por mínimo del histograma":
                resultado = self.seg.segmentacionMinimoHistograma(actual)
                self.setResultado(resultado)
            elif choice == "Contar Objetos":
                self.conteo.conteoCompleto(actual)
            elif choice == "Segmentación Watershed":
                resultado = self.seg.segmentacionWatershed(actual)
                self.setResultado(resultado)
            elif choice == "Segmentación K-means":
                resultado = self.seg.segmentacionKMeans(actual, k=3)
                self.setResultado(resultado)
            elif choice == "Segmentación Mean Shift":
                resultado = self.seg.segmentacionMeanShift(actual)
                self.setResultado(resultado)
            elif choice == "Segmentación GrabCut":
                resultado = self.seg.segmentacionGrabCut(actual)
                self.setResultado(resultado)
        except Exception as e:
            msg.error_message(f"Error en las opciones de color: {str(e)}")
            print(f"Error en las opciones de color: {str(e)}")

    def operaciones_action(self, choice): #Acciones del menú de operaciones
        try:
            actual = self.obtener_imagen_actual()
            if actual is None: 
                msg.alerta_message("No se ha cargado una imagen.")
                return
            
            if self.resultado is not None: #Si hay un resultado, se guarda en la pila de cambios para que no se pierda
                self.cambios.guardar(self.resultado.copy())

            if choice == "Suma":
                resultado = self.op.suma(img1 = self.imagen1, img2 = self.const if self.imagen2 is None else self.imagen2)
                self.setResultado(resultado)
            elif choice == "Resta":
                resultado = self.op.resta(img1 = self.imagen1, img2 = self.const if self.imagen2 is None else self.imagen2)
                self.setResultado(resultado)
            elif choice == "Multiplicación":
                resultado = self.op.multiplicacion(img1 = self.imagen1, img2 = self.const if self.imagen2 is None else self.imagen2)
                self.setResultado(resultado)
            elif choice == "AND" or choice == "OR" or choice == "XOR":
                if self.imagen2 is None:
                    msg.alerta_message("Debe cargar dos imágenes para realizar operaciones lógicas.")
                    return

                resultado = self.op._operacion_logica(self.imagen1, self.imagen2, choice)
                self.setResultado(resultado)
            elif choice == "NOT":
                resultado = self.op.negacion(actual)
                self.setResultado(resultado)
            elif choice == "Ecualizar Uniformemente":
                resultado = self.ec.ecualizar_uniformemente(actual)
                self.setResultado(resultado)
            elif choice == "Ecualización Rayleigh":
                resultado = self.ec.rayleigh(actual)
                self.setResultado(resultado)
            elif choice == "Ecualización hipercúbica":
                resultado = self.ec.hipercubica(actual)
                self.setResultado(resultado)
            elif choice == "Ecualización exponencial":
                resultado = self.ec.exponencial(actual)
                self.setResultado(resultado)
            elif choice == "Ecualización logaritmo hiperbólica":
                resultado = self.ec.logHiperbolica(actual)
                self.setResultado(resultado)
            elif choice == "Expansión":
                resultado = self.ec.expansion(actual)
                self.setResultado(resultado)
            elif choice == "Contracción":
                resultado = self.ec.contraccion(actual)
                self.setResultado(resultado)
            elif choice == "Corrección Gamma":
                resultado = self.ec.correccionGamma(actual, gamma=self.const)
                self.setResultado(resultado)
            elif choice == "Ecualización Adaptativa":
                resultado = self.ec.ecualizacionAdaptativa(actual)
                self.setResultado(resultado)
        except Exception as e:
            msg.error_message(f"Error en las operaciones: {str(e)}")
            print(f"Error al realizar la operación: {str(e)}")

    def filtros_action(self, choice): #Acciones del menú de filtros
        try:
            actual = self.obtener_imagen_actual()
            if actual is None:
                msg.alerta_message("No se ha cargado una imagen.")
                return
            
            if self.resultado is not None: #Si hay un resultado, se guarda en la pila de cambios para que no se pierda
                self.cambios.guardar(self.resultado.copy())

            if choice == "Añadir ruido impulsivo":
                resultado = self.ruido.ruido_salPimienta(actual, p=0.02)
                self.setResultado(resultado)
            elif choice == "Añadir ruido Gaussiano":
                resultado = self.ruido.ruidoGaussiano(actual, desEs = self.sigma)
                self.setResultado(resultado)
            elif choice == "Añadir ruido multiplicativo":
                resultado = self.ruido.ruidoMultiplicativo(actual)
                self.setResultado(resultado)
            elif choice == "Filtro Máximo":
                resultado = self.fPBNL.aplicar_filtro(actual, choice, self.t_kernel)
                self.setResultado(resultado)
            elif choice == "Filtro Mínimo":
                resultado = self.fPBNL.aplicar_filtro(actual, choice, self.t_kernel)
                self.setResultado(resultado)
            elif choice == "Filtro promediador":
                resultado = self.fPBNL.filtro_promediador(actual, ksize = self.t_kernel)
                self.setResultado(resultado)
            elif choice == "Filtro promediador pesado":
                resultado = self.fPBNL.filtro_promediador_pesado(actual, N = self.const)
                self.setResultado(resultado)
            elif choice == "Filtro mediana":
                resultado = self.fPBNL.filtro_mediana(actual, ksize = self.t_kernel)
                self.setResultado(resultado)
            elif choice == "Filtro bilateral":
                resultado = self.fPBNL.filtro_bilateral(actual)
                self.setResultado(resultado)
            elif choice == "Filtro Gaussiano":
                resultado = self.fPBNL.filtro_gaussiano(actual, ksize = self.t_kernel, sigmaX = self.sigma)
                self.setResultado(resultado)
            elif choice == "Filtro de Canny":
                resultado = self.fPA.Canny(actual, sig = self.sigma)
                self.setResultado(resultado)
        except Exception as e:
            msg.error_message(f"Error al aplicar el filtro: {str(e)}")
            print(f"Error al aplicar el filtro: {str(e)}")

    def mascaras_y_operadores_action(self, choice): #Acciones del menú de máscaras y operadores
        try:
            actual = self.obtener_imagen_actual()
            if actual is None:
                msg.alerta_message("No se ha cargado una imagen.")
                return
            
            # La lógica para guardar el estado previo se mueve al popup para Laplace
            # para evitar guardados dobles o innecesarios.
            if choice != "Operador de Laplace" and self.resultado is not None:
                self.cambios.guardar(self.resultado.copy())

            if choice == "Operador de Sobel":
                resultado = self.maskOp.sobel(actual)
                self.setResultado(resultado)
            elif choice == "Operador de Prewitt":
                resultado = self.maskOp.prewitt(actual)
                self.setResultado(resultado)
            elif choice == "Operador de Roberts":
                resultado = self.maskOp.roberts(actual)
                self.setResultado(resultado)
            elif choice == "Operador de Laplace":
                self.elegir_kernel_laplace()
            elif choice == "Mascaras de Kirsch":
                resultado = self.maskOp.kirsch(actual, dir=self.kirsch_dir)
                self.setResultado(resultado)
            elif choice == "Máscaras de Robinson":
                resultado = self.maskOp.robinson(actual)
                self.setResultado(resultado)
            elif choice == "Máscaras de Frei-Chen":
                resultado = self.maskOp.frei_chen(actual)
                self.setResultado(resultado)
        except Exception as e:
            msg.error_message(f"Error al aplicar el operador: {str(e)}")
            print(f"Error al aplicar el operador: {str(e)}")

    def morfologia_action(self, choice): #Acciones del menú de morfología
        try:
            actual = self.obtener_imagen_actual()
            if actual is None:
                msg.alerta_message("No se ha cargado una imagen.")
                return
            
            if self.resultado is not None: #Si hay un resultado, se guarda en la pila de cambios para que no se pierda
                self.cambios.guardar(self.resultado.copy())

            if choice == "Erosión":
                resultado = self.morfologia.erosionCV(actual)
                self.setResultado(resultado)
            elif choice == "Dilatación":
                resultado = self.morfologia.dilatacionCV(actual)
                self.setResultado(resultado)
            elif choice == "Apertura":
                resultado = self.morfologia.aperturaCV(actual)
                self.setResultado(resultado)
            elif choice == "Cierre":
                resultado = self.morfologia.cierreCV(actual)
                self.setResultado(resultado)
            elif choice == "Hit or Miss":
                resultado = self.morfologia.hitOrMissCV(actual)
                self.setResultado(resultado)
            elif choice == "Frontera Interior":
                resultado = self.morfologia.fronteraInteriorCV(actual)
                self.setResultado(resultado)
            elif choice == "Frontera Exterior":
                resultado = self.morfologia.fronteraExteriorCV(actual)
                self.setResultado(resultado)
            elif choice == "Apertura tradicional":
                resultado = self.morfologia.apertura(actual)
                self.setResultado(resultado)
            elif choice == "Cierre tradicional":
                resultado = self.morfologia.cierre(actual)
                self.setResultado(resultado)
            elif choice == "Thinning":
                resultado = self.morfologia.adelgazamiento(actual)
                self.setResultado(resultado)
            elif choice == "Esqueleto":
                resultado = self.morfologia.esqueleto(actual)
                self.setResultado(resultado)
            elif choice == "Tophat":
                resultado = self.morfologia.tophat(actual)
                self.setResultado(resultado)
            elif choice == "Blackhat":
                resultado = self.morfologia.blackhat(actual)
                self.setResultado(resultado)
            elif choice == "Gradiente simétrico":
                resultado = self.morfologia.gradSim(actual)
                self.setResultado(resultado)

        except Exception as e:
            msg.error_message(f"Error al aplicar la morfología: {str(e)}")
            print(f"Error al aplicar la morfología: {str(e)}")

    def elegir_umbral(self): #Popup para elegir el umbral
        actual = self.obtener_imagen_actual()
        if actual is None:
            msg.alerta_message("No se ha cargado una imagen.")
            return
        
        self.ventana_umbral = cusTK.CTkToplevel(self)#Crear ventana emergente
        self.ventana_umbral.title("Ajustar umbral")
        self.ventana_umbral.geometry("300x150")
        self.ventana_umbral.grab_set()  #Hace modal la ventana

        self.label_umbral_popup = cusTK.CTkLabel(self.ventana_umbral, text="Umbral: 127") #Etiqueta
        self.label_umbral_popup.pack(pady=10)

        self.slider_umbral_popup = cusTK.CTkSlider( #Slider
            self.ventana_umbral, from_=0, to=255, command=self.actualizar_umbral_popup
        )
        self.slider_umbral_popup.set(127)
        self.slider_umbral_popup.pack(pady=10)
        
        boton_aplicar = cusTK.CTkButton( #Botón para aplicar
            self.ventana_umbral, text="Aplicar", command=self.aplicar_umbral
        )
        boton_aplicar.pack(pady=10)

    def actualizar_umbral_popup(self, valor): #Función para el popup del umbral
        self.label_umbral_popup.configure(text=f"Umbral: {int(valor)}")

    def aplicar_umbral(self): #Función para aplicar el umbral (solo se usa en el popup)
        actual = self.obtener_imagen_actual()
        umbral = int(self.slider_umbral_popup.get())
        resultado = self.op.umbralizar(actual, umbral)
        self.setResultado(resultado)
        self.ventana_umbral.destroy()
    
    def elegir_kernel_laplace(self):
        """"""
        actual = self.obtener_imagen_actual()
        if actual is None:
            msg.alerta_message("No hay una imagen activa para operar.")
            return

        # Crear ventana emergente
        popup = cusTK.CTkToplevel(self)
        popup.title("Kernel de Laplace")
        popup.geometry("300x150")
        popup.grab_set()  # Hace la ventana modal

        cusTK.CTkLabel(popup, text="Seleccione el tipo de kernel:", font=fuente_global).pack(pady=10)

        # Función interna para aplicar el filtro y cerrar el popup
        def aplicar_laplace(tipo):
            if self.resultado is not None:
                self.cambios.guardar(self.resultado.copy())
            resultado = self.maskOp.laplace(actual, tipo_kernel=tipo)
            self.setResultado(resultado)
            popup.destroy()

        # Botones de selección
        btn4 = cusTK.CTkButton(popup, text="Kernel de 5 valores", command=lambda: aplicar_laplace(4))
        btn4.pack(pady=5, padx=20, fill="x")

        btn8 = cusTK.CTkButton(popup, text="Kernel de 9 valores", command=lambda: aplicar_laplace(8))
        btn8.pack(pady=5, padx=20, fill="x")

    def abrir_imagen(self): #Carga de imágenes
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp")])
            if file_path:
                img = cv2.imread(file_path)
                if img is not None:
                    if self.imagen1 is None:
                        self.imagen1 = img
                    elif self.imagen2 is None:
                        self.imagen2 = img
                    else:
                        msg.alerta_message("Ya se han cargado dos imágenes.")
                        return
                    self.mostrar_imagenes()
        except Exception as e:
            msg.error_message(f"Error al abrir la imagen: {str(e)}")
            print(f"Error al abrir la imagen: {str(e)}")

    def setResultado(self, resultado, esDesCambio = False): #Asigna y muestra el resultado de la operación en el frame de resultados
        self.resultado = resultado #Asignación del resultado
        try:
            if resultado is not None:
                resultado_pil = Image.fromarray(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
                resultado_pil.thumbnail((700, 700))
                tk_resultado = CTkImage(dark_image=resultado_pil, size=resultado_pil.size)
                self.resultadoLabel.configure(image=tk_resultado)
                self.cambios.guardar(resultado.copy()) if not esDesCambio else None 
            else:
                tk_resultado = None
                self.resultadoLabel.configure(image=None)
        except Exception as e:
            msg.error_message(f"Error al mostrar el resultado: {str(e)}")
            print(f"Error al mostrar el resultado: {str(e)}")

    def mostrar_imagenes(self): #Muestra las imágenes en los frames correspondientes
        try:
            if self.imagen1 is not None:
                img_rgb1 = cv2.cvtColor(self.imagen1, cv2.COLOR_BGR2RGB)
                img_pil1 = Image.fromarray(img_rgb1)
                img_pil1.thumbnail((800, 400))
                self.tk_img1 = CTkImage(dark_image=img_pil1, size=img_pil1.size)
                self.image_label1.configure(image=self.tk_img1)
            else:
                self.tk_img1 = None
                self.image_label1.configure(image=None)

            if self.imagen2 is not None:
                img_rgb2 = cv2.cvtColor(self.imagen2, cv2.COLOR_BGR2RGB)
                img_pil2 = Image.fromarray(img_rgb2)
                img_pil2.thumbnail((800, 400))
                self.tk_img2 = CTkImage(dark_image=img_pil2, size=img_pil2.size)
                self.image_label2.configure(image=self.tk_img2)
            else:
                self.tk_img2 = None
                self.image_label2.configure(image=None)
        except Exception as e:
            msg.error_message(f"Error al mostrar las imágenes: {str(e)}")
            print(f"Error al mostrar las imágenes: {str(e)}")

    def guardar_imagen(self):
        try:
            actual = self.obtener_imagen_actual()
            if actual is not None:
                file_path = filedialog.asksaveasfilename( #Elegir la ruta y el nombre del archivo
                    defaultextension=".png",
                    filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("BMP files", "*.bmp")],
                    title="Guardar imagen como"
                )
                if file_path:
                    cv2.imwrite(file_path, actual) #Guardar la imagen usando OpenCV
                return
            else:
                msg.alerta_message("No hay una imagen activa para guardar.")
                return
        except Exception as e:
            msg.error_message(f"Error al guardar la imagen: {str(e)}")
            print(f"Error al guardar la imagen: {str(e)}")
            return
    
    def cerrar_imagen(self):
        try:
            if self.imagen_actual == 1:
                self.imagen1 = None
                self.mostrar_imagenes()
            elif self.imagen_actual == 2:
                self.imagen2 = None
                self.mostrar_imagenes()
            elif self.imagen_actual == 3:
                self.resultado = None
                self.setResultado(None)
        except Exception as e:
            msg.error_message(f"Error al cerrar la imagen: {str(e)}")
            print(f"Error al cerrar la imagen: {str(e)}")

    def setConstantes(self): #Método para ajustar constantes
        popupC = cusTK.CTkToplevel(self)#Crear ventana emergente
        popupC.title("Ajustar constantes")
        popupC.geometry("300x550")
        popupC.grab_set()  #Hace modal la ventana

        def aceptar():
            try:
                kernel = int(entrada1.get())
                c = int(entrada2.get())
                const = float(entrada3.get())
                segmentos = int(entrada4.get())
                desEst = float(entrada5.get())
                k_dir = self.kirsch_dir_map[self.kirsch_dir_var.get()]  #Obtener el valor real de la dirección seleccionada
                h_blockSize = int(entrada7.get())
                h_ksize = int(entrada8.get())
                h_k = float(entrada9.get())
                if kernel % 2 != 1:
                    msg.alerta_message("El tamaño del kernel tiene que ser un número impar.")
                else:
                    self.t_kernel = kernel
                    self.c = c
                    self.const = const
                    self.maxSeg = segmentos
                    self.sigma = desEst
                    self.kirsch_dir = k_dir
                    self.harris_blockSize = h_blockSize
                    self.harris_ksize = h_ksize
                    self.harris_k = h_k
                    popupC.destroy()
            except ValueError:
                msg.alerta_message("Por favor, ingrese solo números.")

        #Elementos de la ventana
        cusTK.CTkLabel(popupC, text="Tamaño del kernel:").pack(pady=(20, 5))
        entrada1 = cusTK.CTkEntry(popupC)
        entrada1.pack(pady=5)
        entrada1.insert(0, str(self.t_kernel))

        cusTK.CTkLabel(popupC, text="C para umbralizacion adaptativa:").pack(pady=5)
        entrada2 = cusTK.CTkEntry(popupC)
        entrada2.pack(pady=5)
        entrada2.insert(0, str(self.c))

        cusTK.CTkLabel(popupC, text="Constante para operaciones aritméticas,\n filtro promediador pesado y corrección gamma:").pack(pady=5)
        entrada3 = cusTK.CTkEntry(popupC)
        entrada3.pack(pady=5)
        entrada3.insert(0, str(self.const))

        cusTK.CTkLabel(popupC, text="Número máximo de segmentos:").pack(pady=5)
        entrada4 = cusTK.CTkEntry(popupC)
        entrada4.pack(pady=5)
        entrada4.insert(0, str(self.maxSeg))

        cusTK.CTkLabel(popupC, text="Sigma:").pack(pady=5)
        entrada5 = cusTK.CTkEntry(popupC)
        entrada5.pack(pady=5)
        entrada5.insert(0, str(self.sigma))

        cusTK.CTkLabel(popupC, text="Dirección del compass de Kirsch:").pack(pady=5)
        direcciones_kirsch = [
            ("Todos", "T"),
            ("Norte", "N"),
            ("Noreste", "NE"),
            ("Este", "E"),
            ("Sureste", "SE"),
            ("Sur", "S"),
            ("Suroeste", "SW"),
            ("Oeste", "W"),
            ("Noroeste", "NW")
        ]
        #Mostrar solo los nombres en el menú
        opciones_mostrar = [nombre for nombre, _ in direcciones_kirsch]
        #Mapeo nombre -> valor
        self.kirsch_dir_map = {nombre: valor for nombre, valor in direcciones_kirsch}
        self.kirsch_dir_var = tk.StringVar(value=opciones_mostrar[0])
        entrada6 = cusTK.CTkOptionMenu(popupC, values=opciones_mostrar, variable=self.kirsch_dir_var)
        entrada6.pack(pady=5)

        cusTK.CTkLabel(popupC, text="Harris - Block Size:").pack(pady=5)
        entrada7 = cusTK.CTkEntry(popupC)
        entrada7.pack(pady=5)
        entrada7.insert(0, str(self.harris_blockSize))

        cusTK.CTkLabel(popupC, text="Harris - Ksize (Sobel):").pack(pady=5)
        entrada8 = cusTK.CTkEntry(popupC)
        entrada8.pack(pady=5)
        entrada8.insert(0, str(self.harris_ksize))

        cusTK.CTkLabel(popupC, text="Harris - k (sensibilidad):").pack(pady=5)
        entrada9 = cusTK.CTkEntry(popupC)
        entrada9.pack(pady=5)
        entrada9.insert(0, str(self.harris_k))

        cusTK.CTkButton(popupC, text="Aceptar", command=aceptar).pack(pady=15)

    def elementoEstructural_action(self):
        eK = EditorKernel(self, self.t_kernel, self.t_kernel)
        self.wait_window(eK)
        self.morfologia.setEE(eK.getKernel())

        print(self.morfologia.kernel)

    def tm_action(self, choice): #Acciones del menú de template matching
        try:
            if self.imagen1 is None or self.imagen2 is None:
                msg.alerta_message("Debe cargar dos imágenes para realizar Template Matching.")
                return
            
            if self.imagen1.shape < self.imagen2.shape:
                msg.alerta_message("La imagen 1 debe ser más grande que la imagen 2 (template).")
                return
            
            if self.resultado is not None: #Si hay un resultado, se guarda en la pila de cambios para que no se pierda
                self.cambios.guardar(self.resultado.copy())

            manual = False
            
            if choice == "Regular":
                mtd = 'TM_SQDIFF'
            elif choice == "Normalizada":
                mtd = 'TM_SQDIFF_NORMED'
            elif choice == "Correlación":
                mtd = 'TM_CCORR'
            elif choice == "Correlación\n Normalizada":
                mtd = 'TM_CCORR_NORMED'
            elif choice == "Coeficientes de\n Correlación":
                mtd = 'TM_CCOEFF'
            elif choice == "Coeficientes de\n Correlación\n Normalizada":
                mtd = 'TM_CCOEFF_NORMED'
            elif choice == "R Manual":
                mtd = 'TM_SQDIFF'
                manual = True
            elif choice == "RN Manual":
                mtd = 'TM_SQDIFF_NORMED'
                manual = True
            elif choice == "C Manual":
                mtd = 'TM_CCORR'
                manual = True
            elif choice == "CN Manual":
                mtd = 'TM_CCORR_NORMED'
                manual = True
            elif choice == "CC Manual":
                mtd = 'TM_CCOEFF'
                manual = True
            elif choice == "CCN Manual":
                mtd = 'TM_CCOEFF_NORMED'
                manual = True

            if not manual:
                resultado = self.tmO.tm_OpenCV(img=self.imagen1, template=self.imagen2, metodo=mtd)
            else:
                resultado = self.tmO.tm_Manual(img=self.imagen1, template=self.imagen2, metodo=mtd)
            self.setResultado(resultado)

        except Exception as e:
            msg.error_message(f"Error al aplicar la Template Matching: {str(e)}")
            print(f"Error al aplicar la Template Matching: {str(e)}")
    
    def configurar_harris_popup(self, es_manual=False):
        """Popup para configurar parámetros de Harris antes de ejecutar"""
        tipo = "Manual" if es_manual else "OpenCV"
        
        popup = cusTK.CTkToplevel(self)
        popup.title(f"Parámetros Harris {tipo}")
        popup.geometry("350x300")
        popup.grab_set()  # Hace la ventana modal
        
        # Valores por defecto
        default_block = 7 if es_manual else 3
        default_ksize = 9 if es_manual else 5
        default_k = 0.04
        
        # Block Size
        cusTK.CTkLabel(popup, text="Block Size (debe ser impar):", font=fuente_global).pack(pady=(20, 5))
        entrada_block = cusTK.CTkEntry(popup, font=fuente_global)
        entrada_block.pack(pady=5)
        entrada_block.insert(0, str(default_block))
        
        # Ksize
        cusTK.CTkLabel(popup, text="Ksize - Sobel (debe ser impar):", font=fuente_global).pack(pady=5)
        entrada_ksize = cusTK.CTkEntry(popup, font=fuente_global)
        entrada_ksize.pack(pady=5)
        entrada_ksize.insert(0, str(default_ksize))
        
        # K
        cusTK.CTkLabel(popup, text="k - Sensibilidad (0.04-0.06):", font=fuente_global).pack(pady=5)
        entrada_k = cusTK.CTkEntry(popup, font=fuente_global)
        entrada_k.pack(pady=5)
        entrada_k.insert(0, str(default_k))
        
        def aplicar_harris():
            try:
                block = int(entrada_block.get())
                ksize = int(entrada_ksize.get())
                k = float(entrada_k.get())
                
                # Validaciones
                if block % 2 == 0 or ksize % 2 == 0:
                    msg.alerta_message("Block Size y Ksize deben ser números impares.")
                    return
                
                if block < 1 or ksize < 1:
                    msg.alerta_message("Block Size y Ksize deben ser mayores a 0.")
                    return
                
                if k <= 0:
                    msg.alerta_message("k debe ser mayor a 0.")
                    return
                
                # Configurar parámetros y ejecutar
                self.detEsq.setParametros(block, ksize, k)
                
                actual = self.obtener_imagen_actual()
                if actual is None:
                    msg.alerta_message("No se ha cargado una imagen.")
                    popup.destroy()
                    return
                
                if self.resultado is not None:
                    self.cambios.guardar(self.resultado.copy())
                
                # Ejecutar el detector correspondiente
                if es_manual:
                    resultado = self.detEsq.harris_manual(actual)
                else:
                    resultado = self.detEsq.harris_opencv(actual)
                
                self.setResultado(resultado)
                popup.destroy()
                
            except ValueError:
                msg.alerta_message("Por favor, ingrese valores numéricos válidos.")
            except Exception as e:
                msg.error_message(f"Error al aplicar Harris: {str(e)}")
                print(f"Error al aplicar Harris: {str(e)}")
        
        # Botones
        btn_frame = cusTK.CTkFrame(popup)
        btn_frame.pack(pady=20)
        
        cusTK.CTkButton(
            btn_frame, 
            text="Aplicar", 
            command=aplicar_harris,
            font=fuente_global,
            width=100
        ).pack(side="left", padx=5)
        
        cusTK.CTkButton(
            btn_frame, 
            text="Cancelar", 
            command=popup.destroy,
            font=fuente_global,
            width=100
        ).pack(side="left", padx=5)
    
    def detection_action(self, choice):
        """Acciones del menú de detección de esquinas y monedas"""
        try:
            actual = self.obtener_imagen_actual()
            if actual is None:
                msg.alerta_message("No se ha cargado una imagen.")
                return
            
            if choice == "Harris (OpenCV)":
                self.configurar_harris_popup(es_manual=False)
            elif choice == "Harris Manual":
                self.configurar_harris_popup(es_manual=True)
            elif choice == "Detectar Monedas":
                self.configurar_monedas_popup()
                
        except Exception as e:
            msg.error_message(f"Error en detección: {str(e)}")
            print(f"Error en detección: {str(e)}")

    def configurar_monedas_popup(self):
        """Popup para configurar parámetros de detección de monedas"""
        popup = cusTK.CTkToplevel(self)
        popup.title("Parámetros Detección de Monedas")
        popup.geometry("350x400")
        popup.grab_set()
        
        # Scale
        cusTK.CTkLabel(popup, text="Escala (0.1 - 1.0):", font=fuente_global).pack(pady=(20, 5))
        entrada_scale = cusTK.CTkEntry(popup, font=fuente_global)
        entrada_scale.pack(pady=5)
        entrada_scale.insert(0, str(self.monedas_scale))
        
        # Min Area
        cusTK.CTkLabel(popup, text="Área mínima:", font=fuente_global).pack(pady=5)
        entrada_min_area = cusTK.CTkEntry(popup, font=fuente_global)
        entrada_min_area.pack(pady=5)
        entrada_min_area.insert(0, str(self.monedas_min_area))
        
        # Canny Low
        cusTK.CTkLabel(popup, text="Canny umbral bajo:", font=fuente_global).pack(pady=5)
        entrada_canny_low = cusTK.CTkEntry(popup, font=fuente_global)
        entrada_canny_low.pack(pady=5)
        entrada_canny_low.insert(0, str(self.monedas_canny_low))
        
        # Canny High
        cusTK.CTkLabel(popup, text="Canny umbral alto:", font=fuente_global).pack(pady=5)
        entrada_canny_high = cusTK.CTkEntry(popup, font=fuente_global)
        entrada_canny_high.pack(pady=5)
        entrada_canny_high.insert(0, str(self.monedas_canny_high))
        
        # Kernel Size
        cusTK.CTkLabel(popup, text="Tamaño kernel (closing):", font=fuente_global).pack(pady=5)
        entrada_kernel = cusTK.CTkEntry(popup, font=fuente_global)
        entrada_kernel.pack(pady=5)
        entrada_kernel.insert(0, str(self.monedas_kernel_size))
        
        def aplicar_deteccion():
            try:
                scale = float(entrada_scale.get())
                min_area = int(entrada_min_area.get())
                canny_low = int(entrada_canny_low.get())
                canny_high = int(entrada_canny_high.get())
                kernel_size = int(entrada_kernel.get())
                
                # Validaciones
                if not (0.1 <= scale <= 1.0):
                    msg.alerta_message("La escala debe estar entre 0.1 y 1.0")
                    return
                
                if min_area < 0:
                    msg.alerta_message("El área mínima debe ser mayor a 0")
                    return
                
                if canny_low >= canny_high:
                    msg.alerta_message("El umbral bajo de Canny debe ser menor al alto")
                    return
                
                if kernel_size < 1:
                    msg.alerta_message("El tamaño del kernel debe ser mayor a 0")
                    return
                
                # Guardar parámetros
                self.monedas_scale = scale
                self.monedas_min_area = min_area
                self.monedas_canny_low = canny_low
                self.monedas_canny_high = canny_high
                self.monedas_kernel_size = kernel_size
                
                # Configurar y ejecutar
                self.detMon.setParametros(scale, min_area, canny_low, canny_high, kernel_size)
                
                actual = self.obtener_imagen_actual()
                if actual is None:
                    msg.alerta_message("No se ha cargado una imagen.")
                    popup.destroy()
                    return
                
                if self.resultado is not None:
                    self.cambios.guardar(self.resultado.copy())
                
                resultado, dinero, num_monedas, detalle = self.detMon.detectar_monedas(actual)
                self.setResultado(resultado)
                popup.destroy()
                
            except ValueError:
                msg.alerta_message("Por favor, ingrese valores numéricos válidos.")
            except Exception as e:
                msg.error_message(f"Error al detectar monedas: {str(e)}")
                print(f"Error al detectar monedas: {str(e)}")
        
        # Botones
        btn_frame = cusTK.CTkFrame(popup)
        btn_frame.pack(pady=20)
        
        cusTK.CTkButton(
            btn_frame, 
            text="Detectar", 
            command=aplicar_deteccion,
            font=fuente_global,
            width=100
        ).pack(side="left", padx=5)
        
        cusTK.CTkButton(
            btn_frame, 
            text="Cancelar", 
            command=popup.destroy,
            font=fuente_global,
            width=100
        ).pack(side="left", padx=5)
    
    def perimetro_action(self, choice): #Acciones del menú de análisis de perímetro
        try:
            actual = self.obtener_imagen_actual()
            if actual is None:
                msg.alerta_message("No se ha cargado una imagen.")
                return
            
            if self.resultado is not None: #Si hay un resultado, se guarda en la pila de cambios para que no se pierda
                self.cambios.guardar(self.resultado.copy())
            
            if choice == "Analizar Perímetro":
                resultado = self.analisisPerimetro.analizar_perimetro(actual)
                self.setResultado(resultado)
            elif choice == "Perímetro Exacto":
                resultado = self.analisisPerimetro.calcular_perimetro_exacto(actual)
                self.setResultado(resultado)
            elif choice == "Perímetro y Área":
                resultado = self.analisisPerimetro.analizar_perimetro_y_area(actual)
                self.setResultado(resultado)
            elif choice == "Perímetro con Aproximación":
                resultado = self.analisisPerimetro.perimetro_con_aproximacion(actual)
                self.setResultado(resultado)
        except Exception as e:
            msg.error_message(f"Error en análisis de perímetro: {str(e)}")
            print(f"Error en análisis de perímetro: {str(e)}")

if __name__ == "__main__":
    app = App()
    app.mainloop()
    