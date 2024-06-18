import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Datos de entrenamiento
x_train = np.array([100, 150, 200, 200, 300, 310, 320, 320, 100, 158, 500, 800, 900, 1000]).reshape(-1, 1)
y_train = np.array([500, 600, 580, 700, 1000, 1000, 1500, 1200, 600, 800, 2000, 2200, 3000, 5000])

# Crear el modelo de regresión lineal
model = LinearRegression()

# Entrenar el modelo
model.fit(x_train, y_train)

# Visualización de los datos
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Predicción de casas")
plt.ylabel('Precio (en dólares)')
plt.xlabel('Área (metros cuadrados)')

# Visualizar la recta de regresión
plt.plot(x_train, model.predict(x_train), color='blue', linewidth=3)
plt.show()

# Predicción del precio de una casa con 500 metros cuadrados
x_i = np.array([[700]])
precio = model.predict(x_i)[0]
print(f"Precio estimado: ${precio:.2f} dólares")
