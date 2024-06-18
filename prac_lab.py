import numpy as np
import matplotlib.pyplot as plt

# Datos de entrenamiento
x_train = np.array([100, 150, 200, 200, 300, 310, 320, 320, 100, 158, 500, 800, 900, 1000])
y_train = np.array([500, 600, 580, 700, 1000, 1000, 1500, 1200, 600, 800, 2000, 2200, 3000, 5000])

# Función para calcular la salida del modelo
def compute_model_output(x, w, b):
    m = x.shape[0]
    f_wb = np.zeros(m)
    for i in range(m):
        f_wb[i] = w * x[i] + b
    return f_wb

# Visualización de los datos
plt.scatter(x_train, y_train, marker='x', c='r')
plt.title("Predicción de casas")
plt.ylabel('Precio (en dólares)')
plt.xlabel('Área (metros cuadrados)')
plt.show()

# Parámetros iniciales del modelo
w_init = 0
b_init = 0

# Función para calcular el costo
def compute_cost(x, y, w, b):
    m = len(x)
    f_wb = compute_model_output(x, w, b)
    total_cost = (1 / (2 * m)) * np.sum(np.square(f_wb - y))
    return total_cost

# Función para calcular el gradiente
def compute_gradient(x, y, w, b):
    m = len(x)
    f_wb = compute_model_output(x, w, b)
    dj_dw = (1 / m) * np.sum((f_wb - y) * x)
    dj_db = (1 / m) * np.sum(f_wb - y)
    return dj_dw, dj_db

# Algoritmo de descenso de gradiente
def gradient_descent(x, y, w_in, b_in, alpha, num_iters):
    w = w_in
    b = b_in
    J_history = []
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        J_history.append(compute_cost(x, y, w, b))
    return w, b, J_history

# Configuración de parámetros del algoritmo de gradiente descendiente
iterations = 100
alpha = 1.0e-6

# Ejecución del descenso de gradiente
w_opt, b_opt, J_history = gradient_descent(x_train, y_train, w_init, b_init, alpha, iterations)

# Visualización del costo durante el entrenamiento
plt.plot(range(iterations), J_history)
plt.title("Costo durante el entrenamiento")
plt.xlabel("Iteraciones")
plt.ylabel("Costo")
plt.show()

# Predicción del precio de una casa con 700 metros cuadrados
x_i = 700
precio = w_opt * x_i + b_opt
print(f"Precio estimado: ${precio:.2f} dólares")
