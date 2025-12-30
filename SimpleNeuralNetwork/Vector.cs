public class Vector(double[] data)
{
    public double[] Data { get; } = data;

    /// <summary>
    /// Hadamard Product
    /// https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
    /// </summary>
    public static Vector operator *(Vector a, Vector b)
    {
        if (a.Data.Length != b.Data.Length)
        {
            throw new ArgumentException("Vectors must be of the same length.");
        }

        double[] output = new double[a.Data.Length];

        for (int i = 0; i < a.Data.Length; i++)
        {
            output[i] = a.Data[i] * b.Data[i];
        }

        return new Vector(output);
    }

    /// <summary>
    /// Hadamard Product
    /// https://en.wikipedia.org/wiki/Hadamard_product_(matrices)
    /// </summary>
    public void Multiply(Vector vector)
    {
        if (this.Data.Length != vector.Data.Length)
        {
            throw new ArgumentException("Vectors must be of the same length.");
        }

        for (int i = 0; i < this.Data.Length; i++)
        {
            this.Data[i] *= vector.Data[i];
        }
    }

    /// <summary>
    /// 1 - [2, 2, 2] = [ -1, -1, -1 ]
    /// </summary>
    public static Vector operator -(int value, Vector vector)
    {
        double[] output = new double[vector.Data.Length];

        for (int i = 0; i < vector.Data.Length; i++)
        {
            output[i] = value - vector.Data[i];
        }

        return new Vector(output);
    }


    /// <summary>
    /// Example: [1, 2, 3] . [4, 5, 6] = 1*4 + 2*5 + 3*6 = 32
    /// </summary>
    public double DotProduct(Vector vector)
    {
        if (this.Data.Length != vector.Data.Length)
        {
            throw new ArgumentException("Vectors must be of the same length.");
        }

        double dotProduct = 0;
        for (int i = 0; i < this.Data.Length; i++)
        {
            dotProduct += this.Data[i] * vector.Data[i];
        }

        return dotProduct;
    }

    /// <summary>
    /// Cosine Similarity between two vectors
    /// https://en.wikipedia.org/wiki/Cosine_similarity
    /// </summary>
    public double CosineProduct(Vector vector)
    {
        double dotProduct = this.DotProduct(vector);
        double magnitudeA = Math.Sqrt(this.DotProduct(this));
        double magnitudeB = Math.Sqrt(vector.DotProduct(vector));
        if (magnitudeA == 0 || magnitudeB == 0)
        {
            return 0;
        }

        return dotProduct / (magnitudeA * magnitudeB);
    }

    /// <summary>
    /// Example: [1, 
    ///           2, 
    ///           3],  . [4, 5, 6] =
    /// 
    ///          [1*4, 1*5, 1*6],
    ///          [2*4, 2*5, 2*6],
    ///          [3*4, 3*5, 3*6]
    /// </summary>
    public Matrix Dot(Vector vector)
    {
        double[][] result = new double[this.Data.Length][];

        for (int i = 0; i < this.Data.Length; i++)
        {
            result[i] = new double[vector.Data.Length];

            for (int j = 0; j < vector.Data.Length; j++)
            {
                result[i][j] = this.Data[i] * vector.Data[j];
            }
        }

        return new Matrix(result);
    }
    
    public Vector ApplyFunctionElementWise(Func<double, double> func)
    {
        double[] output = new double[Data.Length];
        
        for (int i = 0; i < Data.Length; i++)
        {
            output[i] = func(this.Data[i]);
        }

        return new Vector(output);
    }

    public Vector ApplyFunctionElementWise(Vector vector, Func<double, double, double> func)
    {
        double[] output = new double[Data.Length];

        for (int i = 0; i < Data.Length; i++)
        {
            output[i] = func(this.Data[i], vector.Data[i]);
        }

        return new Vector(output);
    }
}
