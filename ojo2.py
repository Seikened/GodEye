import cv2
import face_recognition
import pickle
import os


ruta = os.getcwd()
names_path = os.path.join(ruta,'data' ,'names.pkl')
face_data_path = os.path.join(ruta,'data' ,'faces_data.pkl')


with open(names_path, 'rb') as f:
    nombres_conocidos = pickle.load(f)
with open(face_data_path, 'rb') as f:
    codificaciones_conocidas = pickle.load(f)


cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1080)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while True:
    ret, frame = cap.read()
    if not ret:
        break

    caras_detectadas = face_recognition.face_locations(frame, model='cnn')

    for top, right, bottom, left in caras_detectadas:
        face_frame = frame[top:bottom, left:right]
        face_frame_rgb = cv2.cvtColor(face_frame, cv2.COLOR_BGR2RGB)

        codificaciones_caras = face_recognition.face_encodings(face_frame_rgb, known_face_locations=[(0, face_frame.shape[1], face_frame.shape[0], 0)])

        for codificacion_cara in codificaciones_caras:
            coincidencias = face_recognition.compare_faces(codificaciones_conocidas, codificacion_cara)
            nombre = "Desconocido"
            
            if True in coincidencias:
                primer_coincidencia = coincidencias.index(True)
                nombre = nombres_conocidos[primer_coincidencia]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.putText(frame, nombre, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
