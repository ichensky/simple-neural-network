using System.Text;

public class GnuPlotHelpers 
{
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

    public static string TrainingLosses<T>(IList<T> losses, string title)
    {
        var script = new StringBuilder();
        script.Append(@$"
        set title '{title}'
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

    public static string GeneratorOutput<T>(IList<IList<T>> outputs, string title)
    {
        var transposed = Transpose(outputs);

        var script = new StringBuilder();
        script.Append(@$"
        set title '{title}'

        set palette gray

        plot '-' matrix with image
        ");

        for (int i = 0; i < transposed.Length; i++)
        {
            script.AppendLine(string.Join(' ' , transposed[i]));
        }

        script.AppendLine("e");

        return script.ToString();
    }

    public static T[][] Transpose<T>(IList<IList<T>> data)
    {
        int rows = data.Count;
        int cols = data[0].Count;
        T[][] transposed = new T[cols][];

        for (int i = 0; i < cols; i++)
        {
            transposed[i] = new T[rows];
            for (int j = 0; j < rows; j++)
            {
                transposed[i][j] = data[j][i];
            }
        }

        return transposed;
    }

}