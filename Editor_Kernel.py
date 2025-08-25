import customtkinter as ctk
import numpy as np
import tkinter as tk
from tkinter import simpledialog
import Messages as msg

class EditorKernel(ctk.CTkToplevel):
    def __init__(self, master, filas=3, columnas=3):
        super().__init__(master)
        self.title("Editor de Kernel")

        self.filas = filas
        self.columnas = columnas
        self.kernel = np.ones((filas, columnas), dtype=np.uint8)
        self.cuadriculaEntradas = []

        self.marcoCuadricula = ctk.CTkFrame(self)
        self.marcoCuadricula.pack(padx=10, pady=10)

        self.marcoControles = ctk.CTkFrame(self)
        self.marcoControles.pack(pady=(0, 10))

        self.generarCuadricula()

        #Botones de acción
        ctk.CTkButton(self.marcoControles, text="Llenar con 0", command=lambda: self.rellenarCuadricula(0)).grid(row=0, column=0, padx=5)
        ctk.CTkButton(self.marcoControles, text="Llenar con 1", command=lambda: self.rellenarCuadricula(1)).grid(row=0, column=1, padx=5)
        ctk.CTkButton(self.marcoControles, text="Guardar Kernel", command=self.guardarKernel).grid(row=0, column=2, padx=5)
        ctk.CTkButton(self.marcoControles, text="Cambiar tamaño", command=self.cambiarDimensiones).grid(row=0, column=3, padx=5)

    def generarCuadricula(self):
        for widget in self.marcoCuadricula.winfo_children():
            widget.destroy()

        self.cuadriculaEntradas = []
        for i in range(self.filas):
            fila = []
            for j in range(self.columnas):
                cuadroEntrada = ctk.CTkEntry(self.marcoCuadricula, width=50, justify="center")
                cuadroEntrada.grid(row=i, column=j, padx=2, pady=2)
                cuadroEntrada.insert(0, "0")
                fila.append(cuadroEntrada)
            self.cuadriculaEntradas.append(fila)

    def rellenarCuadricula(self, valor):
        for fila in self.cuadriculaEntradas:
            for cuadro in fila:
                cuadro.delete(0, tk.END)
                cuadro.insert(0, str(valor))

    def cambiarDimensiones(self):
        nuevasFilas = simpledialog.askinteger("Filas", "Número de filas:", initialvalue=self.filas, parent=self)
        nuevasColumnas = simpledialog.askinteger("Columnas", "Número de columnas:", initialvalue=self.columnas, parent=self)
        if nuevasFilas and nuevasColumnas and nuevasFilas > 0 and nuevasColumnas > 0:
            self.filas = nuevasFilas
            self.columnas = nuevasColumnas
            self.generarCuadricula()

    def guardarKernel(self):
        try:
            matriz = []
            for fila in self.cuadriculaEntradas:
                filaNumeros = []
                for cuadro in fila:
                    valor = int(cuadro.get())
                    filaNumeros.append(valor)
                matriz.append(filaNumeros)
            self.kernel = np.array(matriz, dtype=np.uint8)
            self.destroy()  # Cierra la ventana al guardar
        except ValueError:
            msg.alerta_message(title="Advertencia", message="Todos los campos deben ser números enteros.")

    def getKernel(self):
        return self.kernel