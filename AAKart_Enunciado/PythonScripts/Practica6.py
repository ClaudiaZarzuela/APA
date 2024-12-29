from Utils import load_data_csv_multi, one_hot_encoding,drawConfusionMatrix, drawMetrixTable, accuracy, ExportAllformatsMLPSKlearn, WriteStandardScaler
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from MLP import MLP_backprop_predict
import mpl_toolkits.mplot3d
import numpy as np
from sklearn import decomposition
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier


# LEEMOS LOS DATOS, LOS LIMPIAMOS Y SEPARAMOS EN ENTRADAS Y SALIDA -------------------------------------------------------
x, y = load_data_csv_multi("../ML/concatenado.csv")



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
WriteStandardScaler("StandarScalerDataCustom.txt",scaler.mean_,scaler.var_)


#DIVIDIMOS LOS DATOS ------------------------------------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)
categories = ["ACCELERATE", "LEFT_ACCELERATE", "RIGHT_ACCELERATE"]



#ENTRENAMOS EL MODELO CON NUESTRO MLP -------------------------------------------------------------------------------------
y_train_one_hot = one_hot_encoding(y_train, categories)
y_pred = MLP_backprop_predict(X_train,y_train_one_hot,X_test,1,0,2000,0)

# MATRIZ DE CONFUSIÓN
y_pred_mapped = np.array(categories)[y_pred]
drawConfusionMatrix(y_pred_mapped, y_test, categories, "Matriz de confusión de nuestro MLP")

#TABLA DE METRICAS
drawMetrixTable(y_pred_mapped, y_test, categories, "Accuracy de nuestro modelo propio MLP: ", accuracy(y_pred_mapped, y_test))



print("--------------------------------------------------------------------------------------------\n")
# VERSION SKLEARN DEL MLP
clf_0 = MLPClassifier(hidden_layer_sizes= [30,15], activation = 'logistic', alpha=0, max_iter= 500, learning_rate_init= 1, epsilon= 0.12, learning_rate="constant")
clf_0.fit(X_train, y_train_one_hot)
clf_0_Predict = clf_0.predict(X_test)
y_pred_mapped = np.array(categories)[np.argmax(clf_0_Predict, axis=1)]
accu = accuracy(y_pred_mapped, y_test)

#EXPORTAR DATOS PARA SU USO EN UNITY
ExportAllformatsMLPSKlearn(clf_0,X_train, "MLP_SKLearn_Pickle", "MLP_SKLearn_Onix", "MLP_SKLearn_JSON", "MLP_SKLearn.txt")


# MATRIZ DE CONFUSIÓN
drawConfusionMatrix(y_pred_mapped, y_test, categories, "Matriz de confusión del MLP (SKLearn)")

#TABLA DE METRICAS
drawMetrixTable(y_pred_mapped, y_test, categories, "Accuracy del modelo MLPClassifier (SKLearn): ", accu)
print("\nNo se ha podido mejorar el accuracy del modelo cambiando los parametros, pero el max_iter puede reducirse a 500 con los mismos resultados")



print("--------------------------------------------------------------------------------------------\n")
# MODELO KNN DE SKLEARN --------------------------------------------------------------------------------------------------
neigh = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2)
neigh.fit(X_train, y_train)
accu = neigh.score(X_test, y_test)
neighPredict = neigh.predict(X_test)

# MATRIZ DE CONFUSIÓN
drawConfusionMatrix(neighPredict, y_test, categories, "Matriz de confusión del KNN (SKLearn)")

#TABLA DE METRICAS
drawMetrixTable(neighPredict, y_test, categories, "Accuracy del modelo KNN (SKLearn): ", accu)



print("--------------------------------------------------------------------------------------------\n")
# ARBOLES DE DECISION -----------------------------------------------------------------------------------------------------
dtc = tree.DecisionTreeClassifier(random_state=42, criterion= 'entropy', min_samples_split = 3, max_depth = 13 )
dtc = dtc.fit(X_train, y_train)
y_predict = dtc.predict(X_test)
accu = dtc.score(X_test, y_test)

# MATRIZ DE CONFUSIÓN
drawConfusionMatrix(y_predict, y_test, categories, "Matriz de confusión del modelo de Arboles de Decision (SKLearn)")

#TABLA DE METRICAS
drawMetrixTable(y_predict, y_test, categories, "Accuracy del modelo con Arboles de Decision (SKLearn): ", accu)



print("--------------------------------------------------------------------------------------------\n")
# RANDOM FOREST -----------------------------------------------------------------------------------------------------------
rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train)
y_predict = rf.predict(X_test)
accu = rf.score(X_test, y_test)

# MATRIZ DE CONFUSIÓN
drawConfusionMatrix(y_predict, y_test, categories, "Matriz de confusión del Random Forest (SKLearn)")

#TABLA DE METRICAS
drawMetrixTable(y_predict, y_test, categories, "Accuracy del modelo Random Forest (SKLearn): ", accu)
print("\nLos modelos tiene buenas salidas para las clases ACCELERATE y LEFT_ACCELERATE,\npero bajas con la clase RIGHT_ACCELERATE, probablemente porque tiene muy pocas muestras\n")


print("--------------------------------------------------------------------------------------------\n")

