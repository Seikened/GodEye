import cv2
import face_recognition

# Inicia la captura de video
cap = cv2.VideoCapture(0)  # Cambia a 0 si tu cámara es la principal
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)


while True:
    # Captura un fotograma
    ret, frame = cap.read()
    if not ret:
        break  # Si no se captura el fotograma, sal del bucle

    # Encuentra todas las caras en el fotograma
    caras_detectadas = face_recognition.face_locations(frame, model="hog")  # Puedes usar 'cnn' para mayor precisión #hog

    # Dibuja un rectángulo alrededor de cada cara detectada
    for top, right, bottom, left in caras_detectadas:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    # Muestra el resultado
    cv2.imshow('Video', frame)

    # Rompe el bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cuando todo está hecho, libera la captura
cap.release()
cv2.destroyAllWindows()
