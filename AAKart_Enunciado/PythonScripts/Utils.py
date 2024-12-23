from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_classif
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

def cleanData(X, y):
    #relevance = mutual_info_classif(X, y)
    #f_statistic, p_values = f_classif(X, y)
    
    return X, y

def load_data_csv_multi(path,x1_colum,x2_colum,x3_colum,x4_colum,x5_colum,x6_colum,x7_colum,x8_colum,x9_colum,y_colum):
    data = pd.read_csv(path)
    x1 = data[x1_colum].to_numpy()
    x2 = data[x2_colum].to_numpy()
    x3 = data[x3_colum].to_numpy()
    x4 = data[x4_colum].to_numpy()
    x5 = data[x5_colum].to_numpy()
    x6 = data[x6_colum].to_numpy()
    x7 = data[x7_colum].to_numpy()
    x8 = data[x8_colum].to_numpy()
    x9 = data[x9_colum].to_numpy()
    X = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9])
    X = X.T
    y = data[y_colum].to_numpy()

    X, y = cleanData(X, y)
    return X, y