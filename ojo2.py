import cv2
import face_recognition
import pickle

# Carga de datos y modelo
with open('data/names.pkl', 'rb') as f:
    nombres_conocidos = pickle.load(f)
with open('data/faces_data.pkl', 'rb') as f:
    codificaciones_conocidas = pickle.load(f)

# Inicia la captura de video
cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detecta rostros con 'hog'
    caras_detectadas_hog = face_recognition.face_locations(frame, model='hog')

    # Valida cada detección 'hog' con 'cnn' y obtiene codificaciones solo de caras confirmadas
    caras_confirmadas_y_codificadas = []
    for top, right, bottom, left in caras_detectadas_hog:
        face_frame = frame[top:bottom, left:right]
        face_frame_rgb = face_frame[:, :, ::-1]  # Convertir a RGB
        
        # Valida la detección con 'cnn' en la ROI
        caras_detectadas_cnn = face_recognition.face_locations(face_frame_rgb, model='cnn')
        
        if caras_detectadas_cnn:
            # Solo si 'cnn' confirma una cara, obtén las codificaciones
            codificaciones_caras = face_recognition.face_encodings(face_frame_rgb, caras_detectadas_cnn)
            for codificacion_cara in codificaciones_caras:
                caras_confirmadas_y_codificadas.append((codificacion_cara, (top, right, bottom, left)))

    # Procesa cada rostro confirmado y codificado
    for codificacion_cara, (top, right, bottom, left) in caras_confirmadas_y_codificadas:
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
