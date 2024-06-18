import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos desde el archivo CSV
datos = pd.read_csv("shopping.csv")

print(datos.shape)

etiquetas = datos['Revenue']
evidencia = datos.drop(columns=['Revenue'])

# Convertir variables categóricas a numéricas
evidencia['Month'] = evidencia['Month'].astype('category').cat.codes
evidencia['VisitorType'] = evidencia['VisitorType'].astype('category').cat.codes
evidencia['Weekend'] = evidencia['Weekend'].astype(int)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = train_test_split(evidencia, etiquetas, test_size=0.4, random_state=42)
    
modelo = KNeighborsClassifier(n_neighbors=1)
modelo.fit(X_entrenamiento, y_entrenamiento)

# Realizar predicciones
predicciones = modelo.predict(X_prueba)

# Evaluar el modelo
precision = modelo.score(X_prueba, y_prueba) * 100 
reporte = classification_report(y_prueba, predicciones, target_names=['No Compra', 'Compra'], output_dict=True)
sensibilidad = reporte['Compra']['recall'] * 100 
especificidad = reporte['No Compra']['recall'] * 100 

print(f"Precisión del modelo: {precision:.2f}%")
print(f"Sensibilidad: {sensibilidad:.2f}%")
print(f"Especificidad: {especificidad:.2f}%")

correct = (y_prueba == predicciones).sum()
incorrect = (y_prueba != predicciones).sum()

print(f"Correctos: {correct}")
print(f"Incorrectos: {incorrect}")

# Mostrar informe de clasificación
print("\nClassification Report:")
print(classification_report(y_prueba, predicciones, target_names=['No Compra', 'Compra']))

matriz_confusion = confusion_matrix(y_prueba, predicciones)

sns.heatmap(matriz_confusion, annot=True, fmt='d', cmap='Blues', xticklabels=['No Compra', 'Compra'], yticklabels=['No Compra', 'Compra'])
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.title('Matriz de Confusión')
plt.show()
