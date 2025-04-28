#Archivo para mostrar mensajes personalizados de alerta, error y confirmación.
from CTkMessagebox import CTkMessagebox

def alerta_message(message): #Función para mostrar mensajes de alerta
    CTkMessagebox(title="Alerta", message=message, icon="warning")

def error_message(message): #Función para mostrar mensajes de error
    CTkMessagebox(title="Error", message=message, icon="cancel")

def todobien_message(message): #Función para mostrar mensajes de confirmación
    CTkMessagebox(title="Todo salió bien :)", message=message, icon="check")