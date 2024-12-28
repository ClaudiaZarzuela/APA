from Utils import load_data_csv_multi, precission, recall, one_hot_encoding, accuracy
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split
from MLP import MLP_backprop_predict
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

# LEEMOS LOS DATOS, LOS LIMPIAMOS Y SEPARAMOS EN ENTRADAS Y SALIDA -------------------------------------------------------
x, y = load_data_csv_multi("../ML/concatenado.csv")

# APLICAMOS EL PCA PARA REDUCIR DIMENSIONALIDAD Y MOSTRAMOS ---------------------------------------------------------------
#EN 2D
'''
pca = decomposition.PCA(n_components=2)
x = pca.fit_transform(x)
fig = plt.figure(figsize=(8, 6))

classes = np.unique(y)
colors = plt.cm.rainbow(np.linspace(0,1,len(classes)))

for i, class_value in enumerate(classes):
    plt.scatter(
        x[y == class_value, 0],
        x[y == class_value, 1],
        label=f"Clase {class_value}",
        color=colors[i],
    )

plt.title("Proyección PCA (2 Componentes)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend()
plt.grid(True)
plt.show()
'''

#EN 3D
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

plt.cla()
pca = decomposition.PCA(n_components=3)
pca.fit(x)
x = pca.transform(x)

classes = np.unique(y)
colors = plt.cm.rainbow(np.linspace(0,1,len(classes)))

for i, class_value in enumerate(classes):
    ax.scatter(
        x[y == class_value, 0],
        x[y == class_value, 1],
        x[y == class_value, 2],
        label=f"Clase {class_value}",
        color=colors[i],
    )

ax.set_title("Proyección PCA (3 Componentes)")
ax.set_xlabel("Componente 1")
ax.set_ylabel("Componente 2")
ax.set_zlabel("Componente 3")
ax.legend()
plt.show()

#NORMALIZAMOS LA X --------------------------------------------------------------------------------------------------------
scaler = StandardScaler()
x = scaler.fit_transform(x)

# ENTRENAMOS EL MODELO ----------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
categories = ["ACCELERATE", "LEFT_ACCELERATE", "RIGHT_ACCELERATE"]

y_train = one_hot_encoding(y_train, categories)
y_pred = MLP_backprop_predict(X_train,y_train,X_test,1,0,2000,0)
accu = accuracy(y_test, y_pred,categories)
print(accu)

clf_0 = MLPClassifier(hidden_layer_sizes= [30,15], activation = 'logistic', alpha=0, max_iter= 2000, learning_rate_init= 1, epsilon= 0.12)
clf_0.fit(X_train, y_train)
predicted = clf_0.predict(X_test)
predicted = np.argmax(predicted, axis=1)
accu = accuracy(y_test, predicted,categories)
print(accu)

# MATRIZ DE CONFUSIÓN -----------------------------------------------------------------------------------------------------
'''
Una matriz de confusión es una herramienta fundamental para evaluar el rendimiento de un modelo de clasificación.
Sirve para diagnosticar problemas de clasificación y medir la calidad de las predicciones.
'''

#confusionmatrix = metrics.confusion_matrix((y == 0).astype(int), (p == 0).astype(int))
#cm_display0 = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionmatrix, display_labels=['Negativo','Positivo'])
#cm_display0.plot(cmap=plt.cm.Blues)
#plt.title("Matriz de confusión para la predicción del ACCELERATE")
#plt.text(-0.7, 1.7, f'Precision: {precission(p,y):.4f}', va='center', ha='center', color='blue')
#plt.text(2, 1.7, f'Recall: {recall(p,y):.4f}', va='center', ha='center', color='blue')
#plt.text(1, 1.7, f'Accuracy: {accuracy(p,y):.4f}', va='center', ha='center', color='blue')
#plt.show()
