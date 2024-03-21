import cv2
import face_recognition
import pickle
from sklearn.neighbors import KNeighborsClassifier
from datetime import datetime

# Carga de datos y modelo
with open('data/names.pkl', 'rb') as f:
    LABELS = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(FACES, LABELS) 

# Inicia la captura de video
cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Paso 1: Detecta rostros con 'hog'
    caras_detectadas_hog = face_recognition.face_locations(frame, model='hog')

    # Paso 2 y 3: Para cada detección 'hog', valida con 'cnn'
    caras_confirmadas = []
    for top, right, bottom, left in caras_detectadas_hog:
        # Extrae la ROI según la detección 'hog'
        face_frame = frame[top:bottom, left:right]

        # Valida la detección con 'cnn' en la ROI
        caras_detectadas_cnn = face_recognition.face_locations(face_frame, model='cnn')

        # Si 'cnn' encuentra un rostro en la ROI, considera la detección como confirmada
        if caras_detectadas_cnn:
            caras_confirmadas.append((top, right, bottom, left))

    # Paso 4: Procesa cada rostro confirmado
    for top, right, bottom, left in caras_confirmadas:
        face_frame = frame[top:bottom, left:right]
        face_frame = cv2.resize(face_frame, (50, 50)).flatten().reshape(1, -1)

        name = knn.predict(face_frame)[0]  # Predice el nombre

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()