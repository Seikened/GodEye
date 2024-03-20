import cv2



cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("No se puede acceder a la cámara.")
else:
    print("Acceso a la cámara exitoso.")
    # Limpieza
    cap.release()

cv2.destroyAllWindows()