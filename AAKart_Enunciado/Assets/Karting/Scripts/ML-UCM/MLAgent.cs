using System;
using System.Collections;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using UnityEngine;

public class StandarScaler
{
    private float[] mean;
    private float[] std;
    public StandarScaler(string serieliced)
    {
        string[] lines = serieliced.Split("\n");
        string[] meanStr = lines[0].Split(",");
        string[] stdStr = lines[1].Split(",");
        mean = new float[meanStr.Length];
        std = new float[stdStr.Length];
        for (int i = 0; i < meanStr.Length; i++)
        {
            mean[i] = float.Parse(meanStr[i], System.Globalization.CultureInfo.InvariantCulture);
        }

        for (int i = 0; i < stdStr.Length; i++)
        {
            std[i] = float.Parse(stdStr[i], System.Globalization.CultureInfo.InvariantCulture);
            std[i] = Mathf.Sqrt(std[i]);
        }
    }

    // TODO Implement the standar scaler.
    public float[] Transform(float[] a_input)
    {
        float[] result = new float[a_input.Length];
        for(int i = 0; i < a_input.Length; i++)
        {
            result[i] = (a_input[i] - mean[i]) / std[i];
        }
        return result;
    }
}

public class KNNParameters
{
    private List<float[]> trainingData; 
    private List<string> trainingLabels;
    private int k;

    public KNNParameters()
    {
        trainingData = new List<float[]>();
        trainingLabels = new List<string>();
        k = 0;
    }

    public void SetK(int neighbors){ k = neighbors;}
    public int GetK() { return k; }

    public void AddTrainingData(float[] data) { trainingData.Add(data); }
    public List<float[]> GetTrainingData() { return trainingData; }

    public void AddTrainingLabel(string label) {  trainingLabels.Add(label);}
    public List<string> GetTrainingLabel() { return trainingLabels; }
}
public class MLPParameters
{
    List<float[,]> coeficients;
    List<float[]> intercepts;

    public MLPParameters(int numLayers)
    {
        coeficients = new List<float[,]>();
        intercepts = new List<float[]>();
        for (int i = 0; i < numLayers - 1; i++)
        {
            coeficients.Add(null);
        }
        for (int i = 0; i < numLayers - 1; i++)
        {
            intercepts.Add(null);
        }
    }

    public void CreateCoeficient(int i, int rows, int cols)
    {
        coeficients[i] = new float[rows, cols];
    }

    public void SetCoeficiente(int i, int row, int col, float v)
    {
        coeficients[i][row, col] = v;
    }

    public List<float[,]> GetCoeff()
    {
        return coeficients;
    }
    public void CreateIntercept(int i, int row)
    {
        intercepts[i] = new float[row];
    }

    public void SetIntercept(int i, int row, float v)
    {
        intercepts[i][row] = v;
    }
    public List<float[]> GetInter()
    {
        return intercepts;
    }
}

public class MLModelBase
{
    protected int[] indicesToRemove;
    protected StandarScaler standarScaler;

    public MLModelBase(int[] itr, StandarScaler ss)
    {
        indicesToRemove = itr;
        standarScaler = ss;
    }

    public float[] ConvertPerceptionToInput(Perception p, Transform transform)
    {
        Parameters parameters = Record.ReadParameters(9, Time.timeSinceLevelLoad, p, transform);
        float[] input = parameters.ConvertToFloatArrat();
        float[] a_input = input.Where((value, index) => !indicesToRemove.Contains(index)).ToArray();
        a_input = standarScaler.Transform(a_input);
        return a_input;
    }
}

public class KNNModel: MLModelBase
{
    public KNNParameters knnParameters;

    public KNNModel(KNNParameters p, int[] itr, StandarScaler ss) : base(itr, ss) {
        knnParameters = p;
    }

    public float EuclideanDistance(float[] a, float[] b)
    {
        float sum = 0; 
        for(int i = 0; i < a.Length; i++) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }

        return Mathf.Sqrt(sum);
    }

    public Labels Predict(float[] input)
    {
        var distances = new List<(float distance, Labels label)>();

        for (int i = 0; i < knnParameters.GetTrainingData().Count; i++)
        {
            float[] data = knnParameters.GetTrainingData()[i];
            string label = knnParameters.GetTrainingLabel()[i];
            float distance = EuclideanDistance(input, data);
          
            distances.Add((distance, ConvertIndexToLabel(label)));
        }

        var nearestNeighbors = distances.OrderBy(x => x.distance).Take(knnParameters.GetK()).ToList();

        var groupedLabels = nearestNeighbors
            .GroupBy(x => x.label)
            .OrderByDescending(g => g.Count())
            .First();

        Labels predictedLabel = groupedLabels.Key;

        return predictedLabel;
    }

    public Labels ConvertIndexToLabel(string index)
    {
        Labels output = Labels.NONE;
        switch (index)
        {
            case "ACCELERATE":
                output = Labels.ACCELERATE; break;
            case "LEFT_ACCELERATE":
                output = Labels.LEFT_ACCELERATE; break;
            case "RIGHT_ACCELERATE":
                output = Labels.RIGHT_ACCELERATE; break;
        }
        return output;
    }
}
public class MLPModel : MLModelBase
{
    public MLPParameters mlpParameters;
    public MLPModel(MLPParameters p, int[] itr, StandarScaler ss) : base(itr, ss) {
        mlpParameters = p;
    }

    private float sigmoid(float z)
    {
        return 1f / (1f + Mathf.Exp(-z));
    }


    public bool FeedForwardTest(string csv, float accuracy, float aceptThreshold, out float acc)
    {
        Tuple<List<Parameters>, List<Labels>> tuple = Record.ReadFromCsv(csv, true);
        List<Parameters> parameters = tuple.Item1;
        List<Labels> labels = tuple.Item2;
        int goals = 0;
        for(int i = 0; i < parameters.Count; i++)
        {
            float[] input = parameters[i].ConvertToFloatArrat();
            float[] a_input = input.Where((value, index) => !indicesToRemove.Contains(index)).ToArray();
            a_input = standarScaler.Transform(a_input);
            float[] outputs = FeedForward(a_input);
            if(i == 0)
                Debug.Log(outputs[0] + ","+ outputs[1] + "," + outputs[2]);
            Labels label = Predict(outputs);
            if (label == labels[i])
                goals++;
        }

        acc = goals / ((float)parameters.Count);
        
        float diff = Mathf.Abs(acc - accuracy);
        Debug.Log("Accuracy " + acc + " Accuracy espected " + accuracy + " goalds " + goals + " Examples " + parameters.Count + " Difference "+diff);
        return diff < aceptThreshold;
    }

    // TODO Implement FeedForward
    public float[] FeedForward(float[] a_input)
    {
        List<float[,]> thetas = mlpParameters.GetCoeff();
        List<float[]> sesgos = mlpParameters.GetInter();

        float[] _input = a_input;

        for(int layer = 0; layer < thetas.Count; layer++)
        {
            float[] output = new float[sesgos[layer].Length];

            for(int j = 0; j < output.Length; j++) {
                float sum = 0;
                for(int i = 0; i < _input.Length; i++) 
                {
                    float mult = _input[i] * thetas[layer][i,j];
                    sum += mult;
                }
                output[j] = sigmoid(sum + sesgos[layer][j]);
            }
            _input = output;
        }

        return _input;
    }

    //TODO: implement the conversion from index to actions. You may need to implement several ways of
    //transforming the data if you play in different ways. You must take into account how many classes
    //you have used, and how One Hot Encoder has encoded them and this may vary if you change the training
    //data.
    public Labels ConvertIndexToLabel(int index)
    {
        //categories = ["ACCELERATE", "LEFT_ACCELERATE", "RIGHT_ACCELERATE"]
        Labels output = Labels.NONE;
        switch (index)
        {
            case 0:
                output = Labels.ACCELERATE; break;
            case 1:
                output = Labels.LEFT_ACCELERATE; break;
            case 2:
                output = Labels.RIGHT_ACCELERATE; break;
        }
        return output;
    }
    public Labels Predict(float[] output)
    {
        float max;
        int index = GetIndexMaxValue(output, out max);
        Labels label = ConvertIndexToLabel(index);
        return label;
    }

    public int GetIndexMaxValue(float[] output, out float max)
    {
        max = output[0];
        int index = 0;
        for (int i = 1; i < output.Length; i++)
        {
            if (output[i] > max)
            {
                max = output[i];
                index = i;
            }
        }
        return index;
    }
}

public class MLAgent : MonoBehaviour
{
    public enum ModelType { MLP = 0, KNN = 1 }
    public TextAsset text;
    public ModelType model;
    public bool agentEnable;
    public int[] indexToRemove;
    public TextAsset standarScaler;
    public bool testFeedForward;
    public float accuracy;
    public TextAsset trainingCsv;


    private MLPParameters mlpParameters;
    private MLPModel mlpModel;

    private KNNParameters knnParameters;
    private KNNModel knnModel;
    private Perception perception;

    // Start is called before the first frame update
    void Start()
    {

        if (agentEnable)
        {
            string file = text.text;
            if (model == ModelType.MLP)
            {
                mlpParameters = LoadParameters(file);
                StandarScaler ss = new StandarScaler(standarScaler.text);
                mlpModel = new MLPModel(mlpParameters, indexToRemove, ss);
                Debug.Log("Parameters loaded " + mlpParameters);
                if (testFeedForward)
                {
                    float acc;
                    if(mlpModel.FeedForwardTest(trainingCsv.text, accuracy, 0.025f, out acc))
                    {
                        Debug.Log("Test Complete!");
                    }
                    else
                    {
                        Debug.LogError("Error: Accuracy is not the same. Accuracy in C# "+acc + " accuracy in sklearn "+ accuracy);
                    }
                }
            }
            else if(model == ModelType.KNN)
            {
                knnParameters = LoadParametersKNN(file);
                StandarScaler ss = new StandarScaler(standarScaler.text);
                knnModel = new KNNModel(knnParameters, indexToRemove, ss);
            }
           
            perception = GetComponent<Perception>();
        }
    }


    public KartGame.KartSystems.InputData AgentInput()
    {

        Labels label = Labels.NONE;
        switch (model)
        {
            case ModelType.MLP:
                float[] X = mlpModel.ConvertPerceptionToInput(perception, this.transform);
                float[] outputs = this.mlpModel.FeedForward(X);
                label = this.mlpModel.Predict(outputs);
                break;

            case ModelType.KNN:
                float[] X_KNN = knnModel.ConvertPerceptionToInput(perception, this.transform);
                label = this.knnModel.Predict(X_KNN);
                break;

        }
        KartGame.KartSystems.InputData input = Record.ConvertLabelToInput(label);
        return input;
    }

    public static string TrimpBrackers(string val)
    {
        val = val.Trim();
        val = val.Substring(1);
        val = val.Substring(0, val.Length - 1);
        return val;
    }

    public static int[] SplitWithColumInt(string val)
    {
        val = val.Trim();
        string[] values = val.Split(",");
        int[] result = new int[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            values[i] = values[i].Trim();
            if (values[i].StartsWith("'"))
                values[i] = values[i].Substring(1);
            if (values[i].EndsWith("'"))
                values[i] = values[i].Substring(0, values[i].Length - 1);
            result[i] = int.Parse(values[i]);
        }
        return result;
    }

    public static float[] SplitWithColumFloat(string val)
    {
        val = val.Trim();
        string[] values = val.Split(",");
        float[] result = new float[values.Length];
        for (int i = 0; i < values.Length; i++)
        {
            result[i] = float.Parse(values[i], System.Globalization.CultureInfo.InvariantCulture);
        }
        return result;
    }

    public static KNNParameters LoadParametersKNN(string file)
    {
        KNNParameters parameters = new KNNParameters();

        string[] lines = file.Split("\n");
        bool readingXTrain = false;
        bool readingYTrain = false;

        string val = "";
        string name = "";

        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i].Trim();

            if (!string.IsNullOrEmpty(line))
            {
                if (!readingXTrain && !readingYTrain)
                {
                    string[] nameValue = line.Split(":");

                    if (nameValue.Length == 2)
                    {
                        name = nameValue[0].Trim();
                        val = nameValue[1].Trim();

                        if (name == "n_neighbors")
                        {
                            parameters.SetK(int.Parse(val));
                        }
                    }
                }
                if (line.StartsWith("X_train"))
                {
                    readingXTrain = true;
                    readingYTrain = false;
                }
                else if (line.StartsWith("y_train"))
                {
                    readingXTrain = false;
                    readingYTrain = true;
                }
                else if (readingXTrain)
                {
                    float[] dataRow = line.Split(',')
                        .Select(s => float.Parse(s.Replace(',', '.'), CultureInfo.InvariantCulture)) 
                        .ToArray();
                    parameters.AddTrainingData(dataRow);
                }

                else if (readingYTrain)
                {
                    parameters.AddTrainingLabel(line.Trim());
                }
            }
        }

        return parameters;
    }


    public static MLPParameters LoadParameters(string file)
    {
        string[] lines = file.Split("\n");
        int num_layers = 0;
        MLPParameters mlpParameters = null;
        int currentParameter = -1;
        int[] currentDimension = null;
        bool coefficient = false;
        for (int i = 0; i < lines.Length; i++)
        {
            string line = lines[i];
            line = line.Trim();
            if (line != "")
            {
                string[] nameValue = line.Split(":");
                string name = nameValue[0];
                string val = nameValue[1];
                if (name == "num_layers")
                {
                    num_layers = int.Parse(val);
                    mlpParameters = new MLPParameters(num_layers);
                }
                else
                {
                    if (num_layers <= 0)
                        Debug.LogError("Format error: First line must be num_layers");
                    else
                    {
                        if (name == "parameter")
                            currentParameter = int.Parse(val);
                        else if (name == "dims")
                        {
                            val = TrimpBrackers(val);
                            currentDimension = SplitWithColumInt(val);
                        }
                        else if (name == "name")
                        {
                            if (val.StartsWith("coefficient"))
                            {
                                coefficient = true;
                                int index = currentParameter / 2;
                                mlpParameters.CreateCoeficient(currentParameter, currentDimension[0], currentDimension[1]);
                            }
                            else
                            {
                                coefficient = false;
                                mlpParameters.CreateIntercept(currentParameter, currentDimension[1]);
                            }

                        }
                        else if (name == "values")
                        {
                            val = TrimpBrackers(val);
                            float[] parameters = SplitWithColumFloat(val);

                            for (int index = 0; index < parameters.Length; index++)
                            {
                                if (coefficient)
                                {
                                    int row = index / currentDimension[1];
                                    int col = index % currentDimension[1];
                                    mlpParameters.SetCoeficiente(currentParameter, row, col, parameters[index]);
                                }
                                else
                                {
                                    mlpParameters.SetIntercept(currentParameter, index, parameters[index]);
                                }
                            }
                        }
                    }
                }
            }
        }
        return mlpParameters;
    }
}
