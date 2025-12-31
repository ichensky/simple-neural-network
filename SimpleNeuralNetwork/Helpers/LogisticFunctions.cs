namespace SimpleNeuralNetwork.Helpers;

public static class LogisticFunctions
{
    public static double Sigmoid(double value)
    {
        return 1 / (1 + Math.Exp(-value));
    }
}