import cv2
import face_recognition
import pickle
from sklearn.neighbors import KNeighborsClassifier
import os


# Carga de datos y modelo

ruta = os.getcwd()
names_path = os.path.join(ruta,'data' ,'names.pkl')
face_data_path = os.path.join(ruta,'data' ,'faces_data.pkl')


with open(names_path, 'rb') as f:
    LABELS = pickle.load(f)
with open(face_data_path, 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors=5) 
knn.fit(FACES, LABELS) 

# Inicia la captura de video
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    print(f"LLego hasta aqui y el valor de ret es {ret} y el valor de frame es {frame}")
    if not ret:
        break

    # Paso 1: Detecta rostros con 'hog'
    caras_detectadas_hog = face_recognition.face_locations(frame, model='hog')
    print(f"LLego hasta aqui y el valor de caras_detectadas_hog es {caras_detectadas_hog}")
    # Paso 2 y 3: Para cada detección 'hog', valida con 'cnn'
    caras_confirmadas = []
    for top, right, bottom, left in caras_detectadas_hog:
        # Extrae la ROI según la detección 'hog'
        face_frame = frame[top:bottom, left:right]
        print(f"LLego hasta aqui y estoy dentro del for y el valor de face_frame es {face_frame}")

        # Valida la detección con 'cnn' en la ROI
        caras_detectadas_cnn = face_recognition.face_locations(face_frame, model='hog')
        print(f"LLego hasta aqui y el valor de caras_detectadas_cnn es {caras_detectadas_cnn}")

        # Si 'cnn' encuentra un rostro en la ROI, considera la detección como confirmada
        if caras_detectadas_cnn:
            caras_confirmadas.append((top, right, bottom, left))
            

    # Paso 4: Procesa cada rostro confirmado
    for top, right, bottom, left in caras_confirmadas:
        face_frame = frame[top:bottom, left:right]
        face_frame = cv2.resize(face_frame, (50, 50)).flatten().reshape(1, -1)
        name = "Desconocido"  # Define un valor predeterminado para name

        try:
            name = knn.predict(face_frame)[0]  # Intenta la predicción
        except Exception as e:
            print(f"Se produjo un error durante la predicción: {e}")
            # Podrías continuar o hacer algo especial aquí si ocurre una excepción

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
        cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    # El resto del código...


    cv2.imshow('Video', frame)
    print(f"LLego hasta aqui y el valor de frame es {frame}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print("Saliendo...")
cap.release()
cv2.destroyAllWindows()