import cv2
import face_recognition
import numpy as np
from tensorflow.keras.applications.mobilenet import MobileNet, preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# Carga el modelo MobileNet reentrenado para clasificación de género
# Asegúrate de reemplazar 'path/to/your/model.h5' con la ruta real a tu modelo
modelo_genero = load_model('path/to/your/model.h5')

# Inicia la captura de video
cap = cv2.VideoCapture(0)

while True:
    # Captura un fotograma
    ret, frame = cap.read()
    if not ret:
        break

    # Encuentra todas las caras en el fotograma
    caras_detectadas = face_recognition.face_locations(frame)
    for top, right, bottom, left in caras_detectadas:
        # Extrae la cara
        cara = frame[top:bottom, left:right]
        cara = cv2.resize(cara, (224, 224))  # Ajusta el tamaño de la imagen a lo que espera MobileNet
        cara = img_to_array(cara)
        cara = np.expand_dims(cara, axis=0)
        cara = preprocess_input(cara)

        # Realiza la predicción de género
        genero_pred = modelo_genero.predict(cara)
        genero = "Hombre" if genero_pred[0][0] > 0.5 else "Mujer"

        # Dibuja un rectángulo y añade la etiqueta de género
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, genero, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

    # Muestra el resultado
    cv2.imshow('Video', frame)

    # Rompe el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cuando todo está hecho, libera la captura
cap.release()
cv2.destroyAllWindows()
