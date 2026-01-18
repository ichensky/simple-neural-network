namespace SimpleNeuralNetwork.Helpers;

public static class RandomNumberSamplers
{
    /// <summary>
    /// Box-Muller Transform to get a Normal Distribution
    /// https://en.wikipedia.org/wiki/Box–Muller_transform
    /// </summary>
    public static (double z0, double z1) BoxMullerTransform(double mean, double deviation, Random rand) 
    {
        double u1 = rand.NextDouble(); 
        double u2 = rand.NextDouble();

        double part1 = Math.Sqrt(-2.0 * Math.Log(u1));
        double part2 = 2.0 * Math.PI * u2;
        
        double z0 = part1 * Math.Cos(part2);
        double z1 = part1 * Math.Sin(part2);

        return (z0 * deviation + mean, z1 * deviation + mean);
    }
}