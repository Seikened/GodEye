import os
import pickle

# Leamos el archivo
ruta = 'data/names.pkl'
ruta = 'data/faces_data.pkl'

#Abrimos e imprime el contenido del archivo

with open(ruta,'rb') as f:

    contenido = pickle.load(f)
    print(contenido)