# Metodo montecarlo para contar piezas
import random

pesoPromedio = 200
desviacion = pesoPromedio * 0.047

piezasTotales = 0   

for repetir in range(100):
    pesoTotal = 0
    pesoPieza = random.gauss(pesoPromedio, desviacion)
    pesoTotal += pesoPieza
    pesoPrimeraPieza = pesoPieza
    for i in range(1, 19):
        pesoPieza = random.gauss(pesoPromedio, desviacion)
        pesoTotal += pesoPieza
    
    numeroPieza = round(pesoTotal / pesoPrimeraPieza)
    print(f"Peso total: {pesoTotal} Numero de piezas: {numeroPieza}")
    piezasTotales += numeroPieza

# Muestro el total de piezas
print(f"Total de piezas: {piezasTotales}")