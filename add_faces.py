import cv2
import pickle
import face_recognition
import numpy as np
import os

video = cv2.VideoCapture(0)

detector_de_rostros = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

codificaciones_de_las_caras = []
name = input("Enter Your Name: ")

while len(codificaciones_de_las_caras) < 100:
    ret, frame = video.read()

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector_de_rostros.detectMultiScale(gris, 1.3, 5)

    for (x, y, w, h) in faces:
        crop_img = frame[y:y+h, x:x+w, :]
        resized_img = cv2.resize(crop_img, (150, 150))
        resized_img_rgb = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)

        # Obtener codificaciones para cada rostro detectado
        codificaciones = face_recognition.face_encodings(resized_img_rgb)

        if codificaciones:
            codificaciones_de_las_caras.append(codificaciones[0])
        cv2.putText(frame, str(len(codificaciones_de_las_caras)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 50, 255),1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows() 

# Convertir la lista de codificaciones a un arreglo de numpy
codificaciones_de_las_caras = np.asarray(codificaciones_de_las_caras)

# Guardar las codificaciones y los nombres en archivos pickle
ruta = os.getcwd()
names_path = os.path.join(ruta, 'data', 'names.pkl')
face_data_path = os.path.join(ruta, 'data', 'faces_data.pkl')

if not os.path.exists('data'):
    os.makedirs('data')

# Actualizar o crear el archivo de nombres
if os.path.exists(names_path):
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    names.extend([name] * len(codificaciones_de_las_caras))
else:
    names = [name] * len(codificaciones_de_las_caras)
with open(names_path, 'wb') as f:
    pickle.dump(names, f)

# Actualizar o crear el archivo de codificaciones
if os.path.exists(face_data_path):
    with open(face_data_path, 'rb') as f:
        existing_codifications = pickle.load(f)
    codificaciones_de_las_caras = np.vstack((existing_codifications, codificaciones_de_las_caras))
with open(face_data_path, 'wb') as f:
    pickle.dump(codificaciones_de_las_caras, f)
