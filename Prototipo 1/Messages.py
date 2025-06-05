#Archivo para mostrar mensajes personalizados de alerta, error y confirmación.
from CTkMessagebox import CTkMessagebox
fuente = ("Segoe UI", 14)

def alerta_message(message): #Función para mostrar mensajes de alerta
    CTkMessagebox(title="Alerta", message=message, icon="warning", font=fuente)

def error_message(message): #Función para mostrar mensajes de error
    CTkMessagebox(title="Error", message=message, icon="cancel", font=fuente)

def todobien_message(message): #Función para mostrar mensajes de confirmación.
    CTkMessagebox(title="Todo salió bien :)", message=message, icon="check", font=fuente)