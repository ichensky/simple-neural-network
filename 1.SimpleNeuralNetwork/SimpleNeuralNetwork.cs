using static SimpleNeuralNetwork.Helpers.LogisticFunctions;
using static SimpleNeuralNetwork.Helpers.RandomNumberSamplers;
using SimpleNeuralNetwork.Helpers;

namespace SimpleNeuralNetwork;

public class SimpleNeuralNetwork(Random rand, int inputNodesCount, int hiddenNodesCount, int outputNodesCount, double learningRate)
{
    private Matrix weightsInputHidden;

    private Matrix weightsHiddenOutput;

    public void InitializeWeights()
    {
        weightsInputHidden = new Matrix(InitializeWeightsMatrix(rand,hiddenNodesCount, inputNodesCount));
        weightsHiddenOutput = new Matrix(InitializeWeightsMatrix(rand, outputNodesCount, hiddenNodesCount));
    }

    public Vector Query(Vector inputs)
    {
        Vector hiddenInputs = weightsInputHidden.Dot(inputs);
        Vector hiddenOutputs = hiddenInputs.ApplyFunctionElementWise(ActivationFunction);

        Vector finalInputs = weightsHiddenOutput.Dot(hiddenOutputs);
        Vector finalOutputs = finalInputs.ApplyFunctionElementWise(ActivationFunction);

        return finalOutputs;
    }

    public void Train(Vector inputs, Vector targets)
    {
        // Forward pass
        Vector hiddenInputs = weightsInputHidden.Dot(inputs);
        Vector hiddenOutputs = hiddenInputs.ApplyFunctionElementWise(ActivationFunction);

        Vector finalInputs = weightsHiddenOutput.Dot(hiddenOutputs);
        Vector finalOutputs = finalInputs.ApplyFunctionElementWise(ActivationFunction);

        // Calculate output errors
        Vector outputErrors = finalOutputs.ApplyFunctionElementWise(targets, LossFunctionAbsoluteError);
        Vector hiddenErrors = weightsHiddenOutput.Transpose().Dot(outputErrors);

        // Backpropagation and weight updates
        Matrix weightsHiddenOutputDelta = CalculateWeightDeltas(outputErrors, finalOutputs, hiddenOutputs);
        weightsHiddenOutputDelta.Multiply(learningRate);

        Matrix weightsInputHiddenDelta = CalculateWeightDeltas(hiddenErrors, hiddenOutputs, inputs);
        weightsInputHiddenDelta.Multiply(learningRate);

        weightsHiddenOutput.Add(weightsHiddenOutputDelta);
        weightsInputHidden.Add(weightsInputHiddenDelta);
    }

    public Vector BackQuery(Vector targets)
    {
        // Backward pass
        Vector finalOutputs = targets;
        Vector finalInputs = finalOutputs.ApplyFunctionElementWise(ActivationFunctionInverse);

        Vector hiddenOutputs = weightsHiddenOutput.Transpose().Dot(finalInputs);
        hiddenOutputs -= hiddenOutputs.Data.Min();
        hiddenOutputs /= hiddenOutputs.Data.Max();
        hiddenOutputs *= 0.98;
        hiddenOutputs += 0.01;

        Vector hiddenInputs = hiddenOutputs.ApplyFunctionElementWise(ActivationFunctionInverse);

        Vector inputs = weightsInputHidden.Transpose().Dot(hiddenInputs);
        inputs -= inputs.Data.Min();
        inputs /= inputs.Data.Max();
        inputs *= 0.98;
        inputs += 0.01;

        return inputs;
    }

    /// <summary>
    /// Calculates weight deltas for backpropagation
    /// see https://en.wikipedia.org/wiki/Backpropagation
    /// </summary>
    private static Matrix CalculateWeightDeltas(Vector errors, Vector outputs, Vector inputs)
    {
        Vector gradients = 1 - outputs;
        gradients.Multiply(outputs);
        gradients.Multiply(errors);

        Matrix weightDeltas = gradients.Dot(inputs);
        return weightDeltas;
    }


    private static double LossFunctionSquaredError(double output, double target) => Math.Pow(target - output, 2);

    private static double LossFunctionAbsoluteError(double output, double target) => target - output;

    private static double ActivationFunction(double node) => Sigmoid(node);

    private static double ActivationFunctionInverse(double output) => Math.Log(output / (1 - output));

    private static double[][] InitializeWeightsMatrix(Random rand, int rows, int cols)
    {
        // sum of weights(maxDeviation) should be small to prevent saturation of activation function
        double maxDeviation = 1 / Math.Sqrt(rows);

        var enumerator = Random(0, maxDeviation, rand).GetEnumerator();

        double[][] weights = new double[rows][];

        for (int i = 0; i < rows; i++)
        {
            weights[i] = new double[cols];

            for (int j = 0; j < cols; j++)
            {
                enumerator.MoveNext();
                weights[i][j] = enumerator.Current;
            }
        }

        static IEnumerable<double> Random(double mean, double deviation, Random rand)
        {
            while (true)
            {
                var (z0, z1) = BoxMullerTransform(mean, deviation, rand);
                yield return z0;
                yield return z1;
            }
        }

        return weights;
    }
}







