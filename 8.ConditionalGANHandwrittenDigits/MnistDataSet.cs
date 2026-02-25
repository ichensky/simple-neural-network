using System.Collections;
using SkiaSharp;
using TorchSharp;


public class MnistDataSet(MnistImage[] images) 
    : torch.utils.data.Dataset<(int label, torch.Tensor imageValues, torch.Tensor target)>,
    IEnumerable<(int label, torch.Tensor imageValues, torch.Tensor target)>
{
    private readonly MnistImage[] images = images;

    public override long Count => images.Length;

    public IEnumerator<(int label, torch.Tensor imageValues, torch.Tensor target)> GetEnumerator()
    {
        for (long i = 0; i < images.Length; i++)
        {
            yield return GetTensor(i);
        }
    }

    public override (int label, torch.Tensor imageValues, torch.Tensor target) GetTensor(long index)
    {
        int label = images[index].Label;
        torch.Tensor target = torch.zeros([10]);
        target[label] = 1.0f;

        torch.Tensor imageValues = torch.tensor(images[index].Pixels)
            .to_type(torch.ScalarType.Float32)
            .div(255.0f);
        return (label, imageValues, target);
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    public static void SaveImage(torch.Tensor imageTensor, string outputPath)
    {
        var greyChannel = imageTensor.clone()
            .mul(255.0f);

        int width = 28; 
        int height = 28;

        using var bitmap = new SKBitmap(width, height);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                byte grey = (byte)(255 - greyChannel[y * width + x].item<float>());
                bitmap.SetPixel(x, y, new SKColor(grey, grey, grey));
            }
        }

        SaveImage(bitmap, outputPath);
    }


    public static void SaveImage(SKBitmap bitmap, string outputPath)
    {
        using var data = bitmap.Encode(SKEncodedImageFormat.Jpeg, 100);
        using var stream = File.OpenWrite(outputPath);
        data.SaveTo(stream);
    }
}
