import cv2

# Inicializa una lista vacía para almacenar los índices de las cámaras disponibles
camaras_disponibles = []


for indice in range(10):
    cap = cv2.VideoCapture(indice)  
    if cap.isOpened():  # Si se pudo abrir, añade el índice a la lista
        camaras_disponibles.append(indice)
        cap.release()  # No olvides liberar la cámara
    else:
        break  


print(f"Cámaras disponibles: {camaras_disponibles}")
