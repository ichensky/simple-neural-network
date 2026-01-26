using System.Collections;
using System.IO.Compression;
using SkiaSharp;
using TorchSharp;


public class CelebADataSet(string zipFilePath) : torch.utils.data.Dataset<torch.Tensor>
{
    public override long Count => throw new NotImplementedException();

    public override torch.Tensor GetTensor(long index)
    {
        throw new NotImplementedException();
    }

    public IEnumerable<torch.Tensor> GetTensors()
    {
        var images = LoadImagesFromZip();

        foreach (var image in images)
        {
            var pixels = image.Pixels;

            var arr = pixels
                .SelectMany(pixel => new float[] { pixel.Red / 255.0f, pixel.Green / 255.0f, pixel.Blue / 255.0f }).ToArray();

            image.Dispose();

            yield return torch.FloatTensor(arr);
        }
    }

    private IEnumerable<SKBitmap> LoadImagesFromZip()
    {
        using ZipArchive archive = ZipFile.OpenRead(zipFilePath);

        foreach (var file in GetAllJpgEntries(archive))
        {
            using var stream = file.Open();
            SKBitmap bitmap = SKBitmap.Decode(stream);
            yield return bitmap;
        }
    }

    private static IEnumerable<ZipArchiveEntry> GetAllJpgEntries(ZipArchive archive)
    {
        foreach (var entry in archive.Entries)
        {
            if (Path.GetExtension(entry.FullName)
                .Equals(".jpg", StringComparison.OrdinalIgnoreCase))
            {
                yield return entry;
            }
        }
    }
    
    public void SaveImage(torch.Tensor imageTensor, string outputPath)
    {
        var data = imageTensor.data<float>().ToArray();
        int width = 178; // CelebA image width
        int height = 218; // CelebA image height

        using var bitmap = new SKBitmap(width, height);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                int index = (y * width + x) * 3;
                byte red = (byte)(data[index] * 255);
                byte green = (byte)(data[index + 1] * 255);
                byte blue = (byte)(data[index + 2] * 255);
                bitmap.SetPixel(x, y, new SKColor(red, green, blue));
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
