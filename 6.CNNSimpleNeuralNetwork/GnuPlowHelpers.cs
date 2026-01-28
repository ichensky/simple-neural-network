using System.Text;

namespace SimpleNeuralNetworkTorchSharp;

public class GnuPlotHelpers 
{
    public static string TrainingLosses(IList<double> losses)
    {
        var script = new StringBuilder();
        script.Append(@$"
        set title 'Training Loss over Epochs'
        set xlabel 'Iteration'
        set ylabel 'Loss'
        set grid
        plot '-' with points title 'Training Loss'
        ");

        for (int i = 0; i < losses.Count; i++)
        {
            script.AppendLine($"{i} {losses[i]}");
        }

        script.AppendLine("e");

        return script.ToString();
    }

    public static string ShowImage<T>(T[] imageData, string title)
    {
        var matrixSize = (int)Math.Sqrt(imageData.Length);
        var imageData2D = imageData
            .Chunk(matrixSize)
            .Reverse()
            .ToArray();

        var script = new StringBuilder();
        script.Append(@$"
        set title '{title}'

        set palette gray negative

        plot '-' matrix with image
        ");

        for (int i = 0; i < imageData2D.Length; i++)
        {
            script.AppendLine(string.Join(' ' , imageData2D[i]));
        }

        script.AppendLine("e");

        return script.ToString();    
    }

}