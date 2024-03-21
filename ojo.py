import cv2
import face_recognition
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import csv
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

    caras_detectadas = face_recognition.face_locations(frame, model="hog")
    
    for top, right, bottom, left in caras_detectadas:
        face_frame = frame[top:bottom, left:right]
        face_frame = cv2.resize(face_frame, (50, 50)).flatten().reshape(1, -1)

        name = knn.predict(face_frame)[0]
        
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        # Aquí podrías agregar la lógica para registrar la asistencia en un archivo CSV
        

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()