#paquetes para descargar
#pip install numpy matplotlib opencv-python, customtkinter

#Librearias para la interfaz grafica
import customtkinter as cusTK
from customtkinter import CTkImage
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

#Librerias para PDI
from ecualizacion import Ecualizador
from operaciones import Operaciones

cusTK.set_appearance_mode("Dark")  #Configuración inicial de la apariencia
cusTK.set_default_color_theme("blue")

import cv2
import numpy as np

class App(cusTK.CTk):
    def __init__(self):
        super().__init__()

        #Inicialización de la ventana principal
        self.title("Prototipo 1")
        self.geometry(f"{self.winfo_screenwidth()}x{self.winfo_screenheight()}")
        self.minsize(800, 600)
        self.resizable(True, True)

        #Inicialización de variables para la manipulación de imágenes
        self.op = Operaciones()
        self.ec = Ecualizador()
        self.imagen1 = None
        self.imagen2 = None
        self.resultado = None
        self.imagen_actual = 1  #Elige cual es la imagen que se va a operar. Por defecto, operar con la imagen 1

        #Barra superior
        self.top_bar = cusTK.CTkFrame(self, height=50)
        self.top_bar.pack(side="top", fill="x")

        #Menu para archivos
        self.archivos_menu = cusTK.CTkOptionMenu(
            self.top_bar,
            values=["Abrir", "Cerrar"],
            command=self.archivos_action
        )
        self.archivos_menu.set("Archivos")
        self.archivos_menu.pack(side="left", padx=10, pady=10)

        #Menu de operaciones
        self.operaciones_menu = cusTK.CTkOptionMenu(
            self.top_bar,
            values=["Suma", "Resta", "Multiplicación", "Color a Gris", "Umbralizar", "AND", "OR", "XOR", "Ecualizar Uniformemente", "Histograma Imagen 1"],
            command=self.operaciones_action
        )
        self.operaciones_menu.set("Operaciones")
        self.operaciones_menu.pack(side="left", padx=10, pady=10)

        #Menu para seleccionar imagen
        self.selector_menu = cusTK.CTkOptionMenu(
            self.top_bar,
            values=["Imagen 1", "Imagen 2", "Imagen 3 (Resultado)"],
            command=self.cambiar_imagen_actual
        )
        self.selector_menu.set("Elegir imagen activa")
        self.selector_menu.pack(side="left", padx=10, pady=10)

        self.toggle_button = cusTK.CTkButton(self.top_bar, text="Modo oscuro", command=self.toggle_theme)
        self.toggle_button.pack(side="right", padx=10, pady=10)

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

    def toggle_theme(self): #Cambiar entre modo claro y oscuro
        if cusTK.get_appearance_mode() == "Light":
            cusTK.set_appearance_mode("Dark")
            self.toggle_button.configure(text="Modo claro")
        else:
            cusTK.set_appearance_mode("Light")
            self.toggle_button.configure(text="Modo oscuro")

    def cambiar_imagen_actual(self, seleccion):
        if seleccion == "Imagen 1":
            self.imagen_actual = 1
        elif seleccion == "Imagen 2":
            self.imagen_actual = 2
        elif seleccion == "Imagen 3 (Resultado)":
            self.imagen_actual = 3

        nuevasOpciones = ["Suma", "Resta", "Multiplicación", "Color a Gris", "Umbralizar", "AND", "OR", "XOR", "Ecualizar Uniformemente", f"Histograma Imagen {self.imagen_actual}"]

        self.operaciones_menu.configure(values=nuevasOpciones)
        self.operaciones_menu.set("Operaciones")  #Opcional: volver al título inicial

    def obtener_imagen_actual(self):
        if self.imagen_actual == 1:
            return self.imagen1
        elif self.imagen_actual == 2:
            return self.imagen2
        elif self.imagen_actual == 3:
            return self.resultado
        return None

    def archivos_action(self, choice): #Acciones del menú de archivos
        if choice == "Abrir":
            self.abrir_imagen()
        elif choice == "Salir":
            self.quit()

    def operaciones_action(self, choice): #Acciones del menú de operaciones
        print(f"Operación seleccionada: {choice}")
        if choice == "Suma":
            actual = self.obtener_imagen_actual()
            if actual is None: 
                self.errores_message("No se ha cargado una imagen.")
                return
            resultado = self.op.suma(imagen=actual)
            self.resultado = resultado
            self.mostrar_resultado(resultado)
        elif choice == "Resta":
            actual = self.obtener_imagen_actual()
            if actual is None: 
                self.errores_message("No se ha cargado una imagen.")
                return
            resultado = self.op.resta(imagen=actual)
            self.resultado = resultado
            self.mostrar_resultado(resultado)
        elif choice == "Multiplicación":
            actual = self.obtener_imagen_actual()
            if actual is None: 
                self.errores_message("No se ha cargado una imagen.")
                return
            resultado = self.op.multiplicacion(imagen=actual)
            self.resultado = resultado
            self.mostrar_resultado(resultado)
        elif choice == "Color a Gris":
            actual = self.obtener_imagen_actual()
            if actual is None: 
                self.errores_message("No se ha cargado una imagen.")
                return
            resultado = self.op.aGris(imagen=actual)
            self.resultado = resultado
            self.mostrar_resultado(resultado)
        elif choice == "Umbralizar":
            self.elegir_umbral()
        elif choice == "AND" or choice == "OR" or choice == "XOR":
            if self.imagen2 is None:
                self.errores_message("Debe cargar dos imágenes para realizar operaciones lógicas.")
                return
            resultado = self.op._operacion_logica(self.imagen1, self.imagen2, choice)
            self.resultado = resultado
            self.mostrar_resultado(resultado)
        elif choice == "Ecualizar Uniformemente":
            actual = self.obtener_imagen_actual()
            if actual is None:
                self.errores_message("No se ha cargado una imagen.")
                return
            resultado = self.ec.ecualizar_uniformemente(actual)
            self.resultado = resultado
            self.mostrar_resultado(resultado)
        elif choice == f"Histograma Imagen {self.imagen_actual}":
            if self.obtener_imagen_actual() is None:
                self.errores_message("No se ha cargado una imagen.")
                return
            actual = self.obtener_imagen_actual()
            self.op.mostrar_histograma(actual)

    def elegir_umbral(self): #Popup para elegir el umbral
        actual = self.obtener_imagen_actual()
        if actual is None:
            self.errores_message("No se ha cargado una imagen.")
            return

        #Crear ventana emergente
        self.ventana_umbral = cusTK.CTkToplevel(self)
        self.ventana_umbral.title("Ajustar umbral")
        self.ventana_umbral.geometry("300x150")
        self.ventana_umbral.grab_set()  #Hace modal la ventana

        #Etiqueta
        self.label_umbral_popup = cusTK.CTkLabel(self.ventana_umbral, text="Umbral: 127")
        self.label_umbral_popup.pack(pady=10)

        #Slider
        self.slider_umbral_popup = cusTK.CTkSlider(
            self.ventana_umbral, from_=0, to=255, command=self.actualizar_umbral_popup
        )
        self.slider_umbral_popup.set(127)
        self.slider_umbral_popup.pack(pady=10)

        #Botón para aplicar
        boton_aplicar = cusTK.CTkButton(
            self.ventana_umbral, text="Aplicar", command=self.aplicar_umbral
        )
        boton_aplicar.pack(pady=10)

    def actualizar_umbral_popup(self, valor): #Función para el popup del umbral
        self.label_umbral_popup.configure(text=f"Umbral: {int(valor)}")

    def aplicar_umbral(self): #Función para aplicar el umbral (solo se usa en el popup)
        actual = self.obtener_imagen_actual()
        umbral = int(self.slider_umbral_popup.get())
        resultado = self.op.umbralizar(actual, umbral)
        self.mostrar_resultado(resultado)
        self.ventana_umbral.destroy()

    def abrir_imagen(self): #Carga de imágenes
        file_path = filedialog.askopenfilename(filetypes=[("Imágenes", "*.png;*.jpg;*.jpeg;*.bmp")])
        if file_path:
            img = cv2.imread(file_path)
            if img is not None:
                if self.imagen1 is None:
                    self.imagen1 = img
                elif self.imagen2 is None:
                    self.imagen2 = img
                else:
                    self.errores_message("Ya se han cargado dos imágenes.")
                    return
                self.mostrar_imagenes()

    def mostrar_resultado(self, resultado): #Muestra el resultado de la operación en el frame de resultados
        resultado_pil = Image.fromarray(cv2.cvtColor(resultado, cv2.COLOR_BGR2RGB))
        resultado_pil.thumbnail((700, 700))
        self.resultadoLabel.configure(image=CTkImage(dark_image=resultado_pil, size=resultado_pil.size))

    def mostrar_imagenes(self): #Muestra las imágenes en los frames correspondientes
        if self.imagen1 is not None:
            img_rgb1 = cv2.cvtColor(self.imagen1, cv2.COLOR_BGR2RGB)
            img_pil1 = Image.fromarray(img_rgb1)
            img_pil1.thumbnail((800, 400))
            self.tk_img1 = CTkImage(dark_image=img_pil1, size=img_pil1.size)
            self.image_label1.configure(image=self.tk_img1)
        else:
            self.image_label1.configure(image=None)

        if self.imagen2 is not None:
            img_rgb2 = cv2.cvtColor(self.imagen2, cv2.COLOR_BGR2RGB)
            img_pil2 = Image.fromarray(img_rgb2)
            img_pil2.thumbnail((800, 400))
            self.tk_img2 = CTkImage(dark_image=img_pil2, size=img_pil2.size)
            self.image_label2.configure(image=self.tk_img2)
        else:
            self.image_label2.configure(image=None)

    def errores_message(self, message): #Función para mostrar mensajes de error
        tk.messagebox.showinfo("Error", message)

if __name__ == "__main__":
    app = App()
    app.mainloop()