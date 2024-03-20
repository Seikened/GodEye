# import open cv
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(1)

while True:
    ret, img = cap.read()
    if not ret:
        print("No se pudo obtener el frame. Verifica si la cámara está disponible.")
        break  # Sale del bucle si no se pudo leer el frame

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Tecla ESC
        break
    
    
    
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Cargar una imagen desde un archivo
img = cv2.imread('tu_imagen.jpg')  # Asegúrate de reemplazar 'tu_imagen.jpg' con el camino hacia tu archivo de imagen

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow('img', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
