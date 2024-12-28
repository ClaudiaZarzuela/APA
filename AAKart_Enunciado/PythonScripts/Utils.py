from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn import metrics
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
#from skl2onnx import to_onnx
#from onnx2json import convert
import numpy as np
import pandas as pd
import pickle
import json


def ExportONNX_JSON_TO_Custom(onnx_json,mlp):
    graphDic = onnx_json["graph"]
    initializer = graphDic["initializer"]
    s= "num_layers:"+str(mlp.n_layers_)+"\n"
    index = 0
    parameterIndex = 0
    for parameter in initializer:
        s += "parameter:"+str(parameterIndex)+"\n"
        print(parameter["dims"])
        s += "dims:"+str(parameter["dims"])+"\n"
        print(parameter["name"])
        s += "name:"+str(parameter["name"])+"\n"
        print(parameter["doubleData"])
        s += "values:"+str(parameter["doubleData"])+"\n"
        index = index + 1
        parameterIndex = index // 2
    return s

def ExportAllformatsMLPSKlearn(mlp,X,picklefileName,onixFileName,jsonFileName,customFileName):
    with open(picklefileName,'wb') as f:
        pickle.dump(mlp,f)
    
    onx = to_onnx(mlp, X[:1])
    with open(onixFileName, "wb") as f:
        f.write(onx.SerializeToString())
    
    onnx_json = convert(input_onnx_file_path=onixFileName,output_json_path=jsonFileName,json_indent=2)
    
    customFormat = ExportONNX_JSON_TO_Custom(onnx_json,mlp)
    with open(customFileName, 'w') as f:
        f.write(customFormat)
        
def export_to_json(model, filename):
    model_dict = {"num_layers": len(model.coefs_)}
    parameters = []

    for i, (coef, intercept) in enumerate(zip(model.coefs_, model.intercepts_)):
        parameter = {
            "parameter": i,
            "coefficient": {
                "dims": list(coef.shape),
                "values": coef.flatten().tolist()
            },
            "intercepts": {
                "dims": [1, len(intercept)],
                "values": intercept.tolist()
            }
        }
        parameters.append(parameter)

    model_dict["parameters"] = parameters

    with open(filename, 'w') as f:
        json.dump(model_dict, f)
def export_to_txt(model, filename):
    with open(filename, 'w') as f:
        num_layers = len(model.coefs_) + 1
        f.write(f"num_layers:{num_layers}\n")

        parameter_num = 0
        for _, (coefs, intercepts) in enumerate(zip(model.coefs_, model.intercepts_)):

            for param_type, param_values in [('coefficient', coefs), ('intercepts', intercepts)]:
                dims = list(map(str, reversed(param_values.shape)))
                f.write(f"parameter:{parameter_num}\n")
                f.write(f"dims:{dims}\n")
                f.write(f"name:{param_type}\n")
                f.write(f"values:{param_values.flatten().tolist()}\n")
            parameter_num += 1

def cleanData(data):
    data = data.drop(columns= ['karty', 'time'], errors='ignore')
    data = data.dropna()

    #Limpiar salidas que no usamos
    data = data.drop(data[data["action"] == "NONE"].index)
    data = data.drop(data[data["action"] == "BRAKE"].index)
    data = data.drop(data[data["action"] == "LEFT_BRAKE"].index)
    data = data.drop(data[data["action"] == "RIGHT_BRAKE"].index)
    
    return data

def load_data_csv_multi(path):
    data = pd.read_csv(path)
    data = cleanData(data)

    x_data = data.columns[:-1]
    X = data[data.columns[:-1]].to_numpy()
    y = data[data.columns[-1]].to_numpy() 

    return X, y, x_data

def precission(P, Y, categories):
    precisions = {}
    for cls in categories:
        TP = np.sum((P == cls) & (Y == cls))
        FP = np.sum((P == cls) & (Y != cls))
        precisions[cls] = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    return precisions
   

def recall(P, Y, categories):
    recalls = {}
    for cls in categories:
        TP = np.sum((P == cls) & (Y == cls))
        FN = np.sum((P != cls) & (Y == cls))
        recalls[cls] = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    return recalls

def accuracy(P,Y):
	y = np.array(Y)
	p = np.array(P)
	return np.mean(p==y)

def f1_scores(P, Y, categories):
    scores = {}
    for cls in categories:
        TP = np.sum((P == cls) & (Y == cls))
        FP = np.sum((P == cls) & (Y != cls))
        FN = np.sum((P != cls) & (Y == cls))
        recallsCls = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        precissionCls = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        scores[cls] = 2* (precissionCls*recallsCls) /(precissionCls + recallsCls) if (precissionCls + recallsCls)> 0 else 0.0
    return scores

def one_hot_encoding(Y, categories):
    _categories = [categories]
    oneHotEncoder = OneHotEncoder(categories=_categories)
    return  oneHotEncoder.fit_transform(Y.reshape(-1,1)).toarray()



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
def drawMetrixTable(y_pred,y_test, categories, text, accu):
    precisions = precission(y_pred, y_test, categories)
    recalls = recall(y_pred, y_test, categories)
    f1_s = f1_scores(y_pred, y_test, categories)

    support = {cls: np.sum(y_test == cls) for cls in categories}
    metrics_table = pd.DataFrame({
    ' ': categories,
    'precision': [precisions[cls] for cls in categories],
    'recall': [recalls[cls] for cls in categories],
    'f1-Score': [f1_s[cls] for cls in categories],
    'support' : [support[cls] for cls in categories]
    })

    metrics_table_str = metrics_table.to_string(
        index=False,
        header=[' ', 'precision', 'recall', 'f1-score', 'support'], 
        float_format='{:.2f}'.format,
        col_space=12
    )

    print("\n", text, accu, "\n")
    print(metrics_table_str)



def drawConfusionMatrix(y_test, y_pred, categories, model):
    '''
    Una matriz de confusión es una herramienta fundamental para evaluar el rendimiento de un modelo de clasificación.
    Sirve para diagnosticar problemas de clasificación y medir la calidad de las predicciones.
    '''
    confusionmatrix = metrics.confusion_matrix(y_test, y_pred, labels=categories)
    cm_display0 = metrics.ConfusionMatrixDisplay(confusion_matrix=confusionmatrix, display_labels=categories)
    cm_display0.plot(cmap=plt.cm.Blues)
    plt.title(model)
    plt.show()
