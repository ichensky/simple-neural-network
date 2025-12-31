namespace SimpleNeuralNetwork.Helpers;

public class Matrix(double[][] data)
{
    public double[][] Data { get; } = data;


    /// <summary>
    /// Example: [[1, 2],        
    ///           [3, 4]]  T =  
    /// 
    ///        [[1, 3],
    ///         [2, 4]]
    /// </summary>
    public Matrix Transpose()
    {
        int rows = this.Data.Length;
        int cols = this.Data[0].Length;
        double[][] transposed = new double[cols][];

        for (int i = 0; i < cols; i++)
        {
            transposed[i] = new double[rows];

            for (int j = 0; j < rows; j++)
            {
                transposed[i][j] = this.Data[j][i];
            }
        }

        return new Matrix(transposed);
    }

    /// <summary>
    /// Example: [[1, 2],        
    ///           [3, 4]]  . [5, 6] =
    /// 
    ///         [1*5 + 2*6,
    ///          3*5 + 4*6]
    /// </summary>
    public Vector Dot(Vector vector)
    {
        if (this.Data[0].Length != vector.Data.Length)
        {
            throw new ArgumentException("Matrix columns must match vector length for dot product.");
        }

        double[] output = new double[this.Data.Length];

        for (int i = 0; i < output.Length; i++)
        {
            double value = 0;

            for (int j = 0; j < vector.Data.Length; j++)
            {
                value += vector.Data[j] * this.Data[i][j];
            }

            output[i] = value;
        }

        return new Vector(output);
    }


    /// <summary>
    /// Example: [[1, 2],        
    ///           [3, 4]]  + [[5, 6],
    ///                     [7, 8]] =
    /// 
    ///         [[6, 8],
    ///          [10, 12]]     
    /// </summary>
    public void Add(Matrix matrix)
    {
        if (this.Data.Length != matrix.Data.Length || this.Data[0].Length != matrix.Data[0].Length)
        {
            throw new ArgumentException("Matrices must have the same dimensions for addition.");
        }

        for (int i = 0; i < this.Data.Length; i++)
        {
            for (int j = 0; j < this.Data[i].Length; j++)
            {
                this.Data[i][j] += matrix.Data[i][j];
            }
        }
    }

    /// <summary>
    /// Example: [[1, 2],        
    ///           [3, 4]]  * 2 =
    /// 
    ///         [[2, 4],
    ///          [6, 8]]     
    /// </summary>
    public void Multiply(double value)
    {
        for (int i = 0; i < this.Data.Length; i++)
        {
            for (int j = 0; j < this.Data[i].Length; j++)
            {
                this.Data[i][j] *= value;
            }
        }
    }
}