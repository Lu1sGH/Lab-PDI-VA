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
                    "Contar Objetos"],
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
        text = cusTK.CTkLabel(self.top_bar2, text="Componentes de Visión Artificial próximamente...", font=fuente_global)
        text.pack(padx=10, pady=10)
        pass

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
                if kernel % 2 != 1:
                    msg.alerta_message("El tamaño del kernel tiene que ser un número impar.")
                else:
                    self.t_kernel = kernel
                    self.c = c
                    self.const = const
                    self.maxSeg = segmentos
                    self.sigma = desEst
                    self.kirsch_dir = k_dir
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

        cusTK.CTkButton(popupC, text="Aceptar", command=aceptar).pack(pady=15)

    def elementoEstructural_action(self):
        eK = EditorKernel(self, self.t_kernel, self.t_kernel)
        self.wait_window(eK)
        self.morfologia.setEE(eK.getKernel())

        print(self.morfologia.kernel)

if __name__ == "__main__":
    app = App()
    app.mainloop()