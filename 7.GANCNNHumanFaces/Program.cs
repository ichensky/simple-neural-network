using GnuPlotWrapper;
using CNNHumanFaces.Networks;
using TorchSharp;
using static TorchSharp.torch;
using Generator = CNNHumanFaces.Networks.Generator;


var zipPath = @"/home/john/Downloads/img_align_celeba.zip";


var dataset = new CelebA128pxDataSet(zipPath);
var images = dataset.GetTensors()
    .Take(100);


// Save first image to verify loading works
dataset.SaveImage(images.First(), "/home/john/Desktop/RAMtmp/original_image.jpg");


var imageWidth = 128;
var imageHeight = 128;
var imageSize = imageWidth * imageHeight * 3; 



var epochs = 4;
using var realLabels = torch.FloatTensor(new float[] { 1.0f }).reshape(1, 1);
using var fakeLabels = torch.FloatTensor(new float[] { 0.0f }).reshape(1, 1);
var discriminator = new Discriminator();
var generator = new Generator();


int im;
for (int epoch = 0; epoch < epochs; epoch++)
{
    im = -1;
    Console.WriteLine($"Epoch {epoch + 1}/{epochs}");

    foreach (var trainImage in images)
    {
        im++;

        if (im %  (10) == 0)
        {
            Console.WriteLine($"Completed {im} images");
        }

        // Train Discriminator on real data
        using var realData = trainImage;
        using var lossReal = discriminator.Train(realData, realLabels);

        // Train Discriminator on fake data
        using var randomSeedDiscriminator = GenerateRandomSeed(100);
        using var fakeData = generator.forward(randomSeedDiscriminator).detach();
        using var lossFake = discriminator.Train(fakeData, fakeLabels);

        // Train Generator
        using var randomSeedGenerator = GenerateRandomSeed(100);
        using var lossGenerator = generator.Train(discriminator, randomSeedGenerator, realLabels);
    }
}





using var imagePlot = new GnuPlot();
imagePlot.Start();

for(int i =0;i<100;i++)
{
    using var randomSeed = GenerateRandomSeed(100);
    using var outputTensor = generator.forward(randomSeed).detach();
    var output = outputTensor.data<float>().ToArray();

    dataset.SaveImage(outputTensor, $"/home/john/Desktop/RAMtmp/modified_image_{i}.jpg");
}












// Plot Discriminator training losses
using var discriminatorGnuPlotWrapper = new GnuPlot();
discriminatorGnuPlotWrapper.Start();
await discriminatorGnuPlotWrapper.ExecuteAsync(GnuPlotHelpers
    .TrainingLosses(discriminator.TrainingLoss, "Discriminator Training Loss").AsMemory());

// Plot Generator training losses
using var generatorGnuPlotWrapper = new GnuPlot();
generatorGnuPlotWrapper.Start();
await generatorGnuPlotWrapper.ExecuteAsync(GnuPlotHelpers
    .TrainingLosses(generator.TrainingLoss, "Generator Training Loss").AsMemory());



// check correctness
using var outputReal = discriminator.forward( torch.FloatTensor(images.First()));
Console.WriteLine($"Real Data Prediction: {outputReal.item<float>()}");

// TODO: fix GenerateRandomImage
// using var outputFake = discriminator.forward(GenerateRandomImage(imageSize));
// Console.WriteLine($"Fake Data Prediction: {outputFake.item<float>()}");


Console.WriteLine("Exiting...");




static Tensor GenerateRandomImage(int size) => torch.rand(size);

static Tensor GenerateRandomSeed(int size) => torch.randn(size);


