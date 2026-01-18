using System.Text;

namespace SimpleNeuralNetwork.Helpers;

public class GnuPlotHelpers 
{
    public static string ConvertToGrayScaleMatrixImageScript<T>(T[] imageData)
    {
        var matrixSize = (int)Math.Sqrt(imageData.Length);
        var imageData2D = imageData
            .Chunk(matrixSize)
            .Reverse()
            .ToArray();


        var script = new StringBuilder();
        script.Append(@$"
        set pm3d map 

        set palette gray

        splot '-' matrix with image
        ");

        foreach (var row in imageData2D)
        {
            script.AppendLine(string.Join(' ', row));
        }

        script.AppendLine("e");

        return script.ToString();
    }
}