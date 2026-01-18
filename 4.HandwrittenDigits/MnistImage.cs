/// <summary>
/// https://pjreddie.com/projects/mnist-in-csv/
/// </summary>
public class MnistImage
{
    public int Label { get; private set; }

    public byte[] Pixels { get; private set; }

    public MnistImage(string csvData)
    {
        string[] arr = csvData.Split(',');
        Label = int.Parse(arr[0]);
        Pixels = [.. arr.Skip(1).Select(str => byte.Parse(str))];
    }
}