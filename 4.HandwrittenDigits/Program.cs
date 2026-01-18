using GnuPlotWrapper;
using TorchSharp;
using HandwrittenDigits.Networks;
using static TorchSharp.torch;
using Generator = HandwrittenDigits.Networks.Generator;


// var pathData = "../../../../data/mnist_train_100.csv";
var pathData = "/home/john/Downloads/mnist_train.csv/mnist_train.csv";
var lines = await File.ReadAllLinesAsync(pathData);
var images = lines.Select(line => new MnistImage(line))
    .ToArray();
    //  .Take(1_000).ToArray();

var dataset = new MnistDataSet(images);




// using var imagePlot1 = new GnuPlot();
// imagePlot1.Start();

// foreach(var img in dataset)
// {
//     await imagePlot1.ExecuteAsync(GnuPlotHelpers
//         .ShowImage(img.imageValues.data<float>().ToArray(), "image").AsMemory());
//     await Task.Delay(1000);
// }


var epochs = 4;
using var realLabels = torch.FloatTensor(new float[] { 1.0f });
using var fakeLabels = torch.FloatTensor(new float[] { 0.0f });
var discriminator = new Discriminator();
var generator = new Generator();


int im;
for (int epoch = 0; epoch < epochs; epoch++)
{
    im = -1;
    Console.WriteLine($"Epoch {epoch + 1}/{epochs}");

    foreach (var trainImage in dataset)
    {
        im++;
        if (im % (images.Length / 100) == 0)
        {
            Console.WriteLine($"Completed {im} images out of {images.Length}");
        }


        // Train Discriminator on real data
        using var realData = trainImage.imageValues;
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

for(int i =0;i<10;i++)
{
    using var randomSeed = GenerateRandomSeed(100);
    using var outputTensor = generator.forward(randomSeed).detach();
    var output = outputTensor.data<float>().ToArray();

    await imagePlot.ExecuteAsync(GnuPlotHelpers
        .ShowImage(output, "image").AsMemory());
    await Task.Delay(100);
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
using var outputReal = discriminator.forward( torch.FloatTensor(images[0].Pixels));
Console.WriteLine($"Real Data Prediction: {outputReal.item<float>()}");

using var outputFake = discriminator.forward(GenerateRandomImage(784));
Console.WriteLine($"Fake Data Prediction: {outputFake.item<float>()}");


Console.WriteLine("Exiting...");




static Tensor GenerateRandomImage(int size) => torch.rand(size);

static Tensor GenerateRandomSeed(int size) => torch.randn(size);


