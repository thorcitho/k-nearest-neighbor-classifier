import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

csv_file = pd.read_csv("shopping.csv")

labels = csv_file['Revenue']
evidence = csv_file.drop(columns=['Revenue'])

# Convertir las variables categóricas a variables numéricas
evidence = pd.get_dummies(evidence)

# Verificar las primeras filas de etiquetas y evidencia
print(labels[:5])
print(evidence.head())


X_training, X_testing, y_training, y_testing = train_test_split(
    evidence, labels, test_size=0.4
)

model = KNeighborsClassifier(n_neighbors=1)
model.fit(X_training, y_training)

# Evaluación del modelo
accuracy = model.score(X_testing, y_testing)
print(f"Accuracy: {accuracy * 100:.2f}")

predictions = model.predict(X_testing)

correct = (y_testing == predictions).sum()
incorrect = (y_testing != predictions).sum()
total = len(predictions)

print(f"Resultados del modelo {type(model).__name__}")
print(f"Correctos: {correct}")
print(f"Incorrectos: {incorrect}")
print(f"Accuracy: {100 * correct / total:.2f}%")