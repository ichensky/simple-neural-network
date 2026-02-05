using System.Collections;
using System.IO.Compression;
using SkiaSharp;
using TorchSharp;


public class CelebA128pxDataSet(string zipFilePath) : torch.utils.data.Dataset<torch.Tensor>
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
            var pixels = CropTo128px(image);

            var red = pixels.Select(p => (float)p.Red).ToArray();
            var green = pixels.Select(p => (float)p.Green).ToArray();
            var blue = pixels.Select(p => (float)p.Blue).ToArray();

            image.Dispose();

            var stacked = torch.stack(
            [
                torch.FloatTensor(red),
                torch.FloatTensor(green),
                torch.FloatTensor(blue)
            ]);

            yield return stacked.view(1, 3, 128, 128).div(255.0f);
        }
    }

    private IEnumerable<SKColor> CropTo128px(SKBitmap skBitmap)
    {
        int cropSize = 128;

        int xOffset = (skBitmap.Width - cropSize) / 2;
        int yOffset = (skBitmap.Height - cropSize) / 2;

        for (int y = 0; y < 128; y++)
        {
            for (int x = 0; x < 128; x++)
            {
                yield return skBitmap.GetPixel(x + xOffset, y + yOffset);
            }
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
        var squeezed = imageTensor.clone()
            .mul(255.0f)
            .squeeze();
        var channels = squeezed.unbind();
        var redChannel = channels[0];
        var greenChannel = channels[1];
        var blueChannel = channels[2];

        // CelebA image width
        int width = 128; 
        // CelebA image height
        int height = 128;

        using var bitmap = new SKBitmap(width, height);
        for (int y = 0; y < height; y++)
        {
            for (int x = 0; x < width; x++)
            {
                byte red = (byte)redChannel[y][x].item<float>();
                byte green = (byte)greenChannel[y][x].item<float>();
                byte blue = (byte)blueChannel[y][x].item<float>();
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
