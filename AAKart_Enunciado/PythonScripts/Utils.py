from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import OneHotEncoder
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

def precission(P, Y):
    TP = np.sum((P == 0) & (Y == 0))
    FP = np.sum((P == 0) & (Y != 0))

    return (TP / (TP + FP))

def recall(P, Y):
    TP = np.sum((P == 0) & (Y == 0))
    FN = np.sum((P != 0) & (Y == 0))

    return (TP / (TP + FN))

def one_hot_encoding(Y, categories):
    _categories = [categories]
    oneHotEncoder = OneHotEncoder(categories=_categories)
    return  oneHotEncoder.fit_transform(Y.reshape(-1,1)).toarray()

def accuracy(P,Y, categories):
	p = np.array(P)
	y = np.array(np.array(categories)[Y])
	return np.mean(p==y)