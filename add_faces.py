import cv2
import pickle
import numpy as np
import os


# Inicialización de la cámara
video=cv2.VideoCapture(1) # el argumento es el índice de la cámara

detector_de_rostros=cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')

datos_de_las_caras=[]

# Contador para almacenar 100 caras
i=0

name=input("Enter Your Name: ")


while True:
    #Captura un frame de la cámara y se almacena en la variable frame y ret (que es un booleano que indica si se pudo capturar el frame o no)
    ret,frame=video.read()
    # Se convierte el frame a escala de grises gracias a la función cvtColor
    gris=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Se detectan los rostros con el método detectMultiScale que recibe como argumentos la imagen en escala de grises, el factor de escala y el número de vecinos (osea, cuántos vecinos deben estar cerca para que sea considerado un rostro)
    faces=detector_de_rostros.detectMultiScale(gris, 1.3 ,5)
    # Recorre cada rostro detectado y los dibuja en el frame
    for (x,y,w,h) in faces:
        # Aqui corta el frame para extraer el rostro
        crop_img=frame[y:y+h, x:x+w, :]
        # Aqui redimensiona el rostro a 50x50 (para que sea igual a los datos de entrenamiento)
        resized_img=cv2.resize(crop_img, (50,50))
        
        #  Se verifica si se han recolectado menos de 100 rostros y si el contador i es múltiplo de 10. Esta condición ayuda a espaciar la recolección de datos de rostros para no capturar imágenes demasiado similares consecutivamente.
        if len(datos_de_las_caras)<=100 and i%10==0:
            datos_de_las_caras.append(resized_img)
        
        # Se incrementa el contador por cada rostro detectado
        i=i+1
        
        # .putText es un método que permite escribir texto en un frame. En este caso, se escribe el número de rostros detectados en la esquina superior izquierda del frame.
        cv2.putText(frame, str(len(datos_de_las_caras)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
        # .rectangle es un método que permite dibujar un rectángulo en un frame. En este caso, se dibuja un rectángulo alrededor del rostro detectado.
        cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
    # Muestra el frame en una ventana con el nombre "Frame"
    cv2.imshow("Frame",frame)
    
    # Si se presiona la tecla "q" o si se han recolectado 100 rostros, se rompe el ciclo
    k=cv2.waitKey(1)
    if k==ord('q') or len(datos_de_las_caras)==100:
        break
video.release()
cv2.destroyAllWindows()

# Convierte la lista de datos de las caras a un arreglo de numpy
datos_de_las_caras=np.asarray(datos_de_las_caras)
# Redimensiona el arreglo a 100 filas y el número de columnas que sea necesario
datos_de_las_caras=datos_de_las_caras.reshape(100, -1)




# Guarda los datos recolectados en un archivo .pkl

# Aqui se verifica si el archivo names.pkl ya existe. Si no existe, se crea una lista con 100 nombres iguales al nombre ingresado por el usuario. 
# Si ya existe, se carga la lista y se le añaden 100 nombres iguales al nombre ingresado por el usuario.
if 'names.pkl' not in os.listdir('data/'):
    names=[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)
else:
    with open('data/names.pkl', 'rb') as f:
        names=pickle.load(f)
    names=names+[name]*100
    with open('data/names.pkl', 'wb') as f:
        pickle.dump(names, f)

# Aqui se verifica si el archivo faces_data.pkl ya existe. Si no existe, se guarda el arreglo de datos recolectados.
if 'faces_data.pkl' not in os.listdir('data/'):
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(datos_de_las_caras, f)
else:
    with open('data/faces_data.pkl', 'rb') as f:
        faces=pickle.load(f)
    faces=np.append(faces, datos_de_las_caras, axis=0)
    with open('data/faces_data.pkl', 'wb') as f:
        pickle.dump(faces, f)