from Utils import load_data_csv_multi, precission, recall, one_hot_encoding, accuracy
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from MLP import MLP_backprop_predict
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier

# LEEMOS LOS DATOS, LOS LIMPIAMOS Y SEPARAMOS EN ENTRADAS Y SALIDA -------------------------------------------------------
x, y, x_data= load_data_csv_multi("../ML/concatenado.csv")
x_columns = x_data # Guardamos las columnas originales para luego mostrar las predicciones de sklearn



#HACEMOS UNA COPIA DE X PARA REALIZAR EL PCA SIN AFECTAR A LOS DATOS ORIGINALES
x_pca = x.copy()

# APLICAMOS EL PCA PARA REDUCIR DIMENSIONALIDAD Y MOSTRAMOS ---------------------------------------------------------------
#EN 2D
pca = decomposition.PCA(n_components=2)
x_pca_2D = pca.fit_transform(x_pca)
fig = plt.figure(figsize=(8, 6))

classes = np.unique(y)
colors = plt.cm.rainbow(np.linspace(0,1,len(classes)))

for i, class_value in enumerate(classes):
    plt.scatter(
        x_pca_2D[y == class_value, 0],
        x_pca_2D[y == class_value, 1],
        label=f"Clase {class_value}",
        color=colors[i],
    )

plt.title("Proyección PCA (2 Componentes)")
plt.xlabel("Componente 1")
plt.ylabel("Componente 2")
plt.legend()
plt.grid(True)
plt.show()

#EN 3D
'''
fig = plt.figure(1, figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

plt.cla()
pca = decomposition.PCA(n_components=3)
x_pca_3d = pca.fit_transform(x_pca)

classes = np.unique(y)
colors = plt.cm.rainbow(np.linspace(0,1,len(classes)))

for i, class_value in enumerate(classes):
    ax.scatter(
        x_pca_3d[y == class_value, 0],
        x_pca_3d[y == class_value, 1],
        x_pca_3d[y == class_value, 2],
        label=f"Clase {class_value}",
        color=colors[i],
    )

ax.set_title("Proyección PCA (3 Componentes)")
ax.set_xlabel("Componente 1")
ax.set_ylabel("Componente 2")
ax.set_zlabel("Componente 3")
ax.legend()
plt.show()
'''

#NORMALIZAMOS LA X --------------------------------------------------------------------------------------------------------
scaler = StandardScaler()
x = scaler.fit_transform(x)


#DIVIDIMOS LOS DATOS ------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
categories = ["ACCELERATE", "LEFT_ACCELERATE", "RIGHT_ACCELERATE"]


#ENTRENAMOS EL MODELO CON NUESTRO MLP -------------------------------------------------------------------------------------
y_train_one_hot = one_hot_encoding(y_train, categories)
y_pred = MLP_backprop_predict(X_train,y_train_one_hot,X_test,1,0,2000,0)
accu = accuracy(y_test, y_pred,categories)
print("\nAccuracy de nuestro MLP: ", accu)



print("--------------------------------------------------------------------------------------------\n")
# VERSION SKLEARN DEL MLP
'''
(solver: algoritmo usado para optimizar los pesos de la red {lbfgs: para pequeños datasets / sgd: para datasets grandes / adam (por defecto): generalmente recomendado})
(learning_rate: (se usa solo si solver = sgd) controla como se ajustan los pesos durante el entrenamiento {constant: fija / invscaling: disminuye la tasa segun avanza
   el entrenamiento / adaptive: fija, pero reduce la tasa si el rendimiento no mejora})
(batch_size: num de muestras usadas para actualizar los pesos en cada iteracion, valores bajos son +ruidosos pero converjen rapido y grandes son +estables pero requieren +memoria)
(tol: umbral tolerancia para la convergencia que detiene el entrenamiento si la mejora relativa de la funcion de perdida es menor que este valor, valores bajos hacen un modelo
   mas preciso pero que requiere mas tiempo de entrenamiento y altos pueden detener el modelo antes de haber alcanzado el optimo)
(random_state: fija la semilla para inicializar los pesos y garantizar reproducibilidad, no afecta a la calidad del modelo pero asegura resultados consistentes)

hidden_layer_sizes: dupla de numero de neuronas por capa oculta, +neuronas = aprende patrones +complejos pero puede causar overfitting mucho tiempo de entrenamiento
activation: funcion de activacion para las neuronas de las capas ocultas, afecta la capacidad del modelo para aprender relaciones no lineales{identity: sin funcion /
   logistic: sigmoide, util para problemas binarios / tanh: tangente hiperbolica, captura relaciones no lineales mas complejas / relu (por defecto): unidad lineal 
   rectificada, eficiente y comunmente usada}
alpha: parametro regularizacion L2 que penaliza pesos grandes y previene el overfitting, valores altos reducen complejidad del modelo y bajos pueden permitir overfitting
max_iter: num maximo de iteraciones para entrenar el modelo
learning_rate_init: (se usa si solver = sgd o solver = adam) valor inicial de la tasa de aprendizaje (valores tipicos 0.0001-0.1), valores altos permiten ajustes rapidos pero 
   pueden causar inestabilidad y bajos pueden ralentizar el aprendizaje y provocar que no converja
epsilon: establece umbral de tolerancia que detiene el entrenamiento si el cambio absoluto en la funcion de perdida entre dos iteraciones es menor que este valor (por defecto 1e-8),
    valores altos pueden detener el modelo antes de haber alcanzado el optimo, y bajos pueden mejorar precision a costa de ralentizar entrenamiento y posible sobreajuste
'''
clf_0 = MLPClassifier(hidden_layer_sizes= [30,15], activation = 'logistic', alpha=0, max_iter= 500, learning_rate_init= 1, epsilon= 0.12)
clf_0.fit(X_train, y_train)
accu = clf_0.score(X_test, y_test)
print("MLP de SKLearn: " + str(accu))

print("Importancia de las categorías en el modelo MLPClassifier")
result = permutation_importance(clf_0, X_test, y_test, n_repeats=10, random_state=0)
importance_df = pd.DataFrame(result.importances_mean, index= x_columns)
print(importance_df.head(10))
print("\nAccuracy del modelo MLPClassifier (SKLearn): ", accu)



print("--------------------------------------------------------------------------------------------\n")
# MODELO KNN DE SKLEARN --------------------------------------------------------------------------------------------------
neigh = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2)
neigh.fit(X_train, y_train)
accu = neigh.score(X_test, y_test)
print("KNN de SKLearn: " + str(accu))



print("--------------------------------------------------------------------------------------------\n")
# ARBOLES DE DECISION -----------------------------------------------------------------------------------------------------
dtc = tree.DecisionTreeClassifier(random_state=40, criterion= 'entropy', min_samples_split = 3, max_depth = 13 )
dtc = dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)
tree.plot_tree(dtc)
accu = dtc.score(X_test, y_test)

print("Importancia de las categorías en el modelo con Arboles de Decision")
features = pd.DataFrame(dtc.feature_importances_, index = x_columns)
print(features.head(10))

print("\nAccuracy del modelo con Arboles de Decision (SKLearn): ", accu)
print(classification_report(y_test, y_predict))



print("--------------------------------------------------------------------------------------------\n")
# RANDOM FOREST -----------------------------------------------------------------------------------------------------------
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_predict = rf.predict(X_test)
accu = rf.score(X_test, y_test)

'''
Precision: Proporción de predicciones correctas sobre el total de predicciones hechas para una clase.
           True Positives (TP) / (Ture Positives (TP) + False Positives (FP))
           Un valor alto indica que las predicciones positivas son generalmente correctas.

Recall: Proporción de ejemplos de una clase que el modelo pudo identificar correctamente.
        True Positives (TP) / ( True Positives (TP) + False Negatives (FN))
        Un valor alto indica que el modelo detecta la mayoría de los ejemplos de una clase.

F1-score: Media armónica entre precisión y recall.
          2*((Precision*Recall)/(Precision+Recall))
          Resume el balance entre precisión y recall. Es útil cuando las clases están desbalanceadas.

Support: Número total de muestras reales para cada clase en el conjunto de datos de prueba.
'''
print("Importancia de las categorías en el modelo Random Forest")
features = pd.DataFrame(rf.feature_importances_, index = x_columns)
print(features.head(10))

print("\nAccuracy del modelo Random Forest (SKLearn): ", accu)
print(classification_report(y_test, y_predict))
print("\nLos modelos tiene buenas salidas para las clases ACCELERATE y LEFT_ACCELERATE,\npero bajas con la clase RIGHT_ACCELERATE, probablemente porque tiene muy pocas muestras\n")



print("--------------------------------------------------------------------------------------------\n")
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
