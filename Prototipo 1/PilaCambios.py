#Clase para poder deshacer cambios en la imagen resultado.
import Messages as msg

class PilaCambios:
    def __init__(self):
        self.pila = [None]

    def guardar(self, cambio): #Agregar cambios a la pila.
        try:
            self.pila.append(cambio)
        except Exception as e:
            msg.error_message(f"Ha ocurrido un error al agregar un cambio a la pila: {str(e)}")

    def deshacer(self): #Regresar al último cambio hecho
        try:
            if len(self.pila) > 2:
                self.pila.pop() #Sin esta linea literalmente se regresa al cambio altual.
                return self.pila.pop()
            else:
                return None
        except Exception as e:
            msg.error_message(f"Ha ocurrido un error al recuperar el último cambio a la pila: {str(e)}")
            return None