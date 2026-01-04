using GnuPlotWrapper;
using SimpleNeuralNetworkTorchSharp;
using TorchSharp;

var pathData = "../../../../data/mnist_train_100.csv";
// var pathData = "/home/john/Downloads/mnist_train.csv/mnist_train.csv";
var lines = await File.ReadAllLinesAsync(pathData);
var images = lines.Select(line => new MnistImage(line))
    .ToArray();
    // .Take(000).ToArray();

// Split into training and test sets
var numberOfTestImages = 1_0;
var numberOfTrainImages = images.Length - numberOfTestImages;
var imagesTrain = images.Take(numberOfTrainImages).ToArray();
var imagesTest = images.Skip(numberOfTrainImages).ToArray();

// var grayScaleMatrixImage = GnuPlowHelpers
//     .ConvertToGrayScaleMatrixImageScript(imagesTrain.First().Pixels);

// using var gnuPlotWrapper = new GnuPlot();
// gnuPlotWrapper.Start();
// await gnuPlotWrapper.ExecuteAsync(grayScaleMatrixImage.AsMemory());


const int epochs = 3;
var classifier = new Classifier();
var trainDataset = new MnistDataSet(imagesTrain);

int counter = 0;
for (int epoch = 0; epoch < epochs; epoch++)
{
    Console.WriteLine($"Epoch {epoch + 1}/{epochs}");
    counter = 0;

    foreach (var (label, imageValues, target) in trainDataset)
    {
        counter++;
        if (counter % 1000 == 0)
        {
            Console.WriteLine($"  Training image {counter}/{trainDataset.Count}");
        }
        classifier.Train(imageValues, target);
    }
}

var trainingLosses = classifier.GetTrainingLosses();

using var gnuPlotWrapper = new GnuPlot();
gnuPlotWrapper.Start();
await gnuPlotWrapper.ExecuteAsync(GnuPlotHelpers.TrainingLosses(trainingLosses).AsMemory());


int score = 0;
var testDataset = new MnistDataSet(imagesTest);
foreach (var (label, imageValues, target) in testDataset)
{
    torch.Tensor output = classifier
        .forward(input: imageValues)
        .detach();

    float[] result = [.. output.data<float>()];

    var ordered = output.data<float>().Select((value, index) => (value, index))
        .OrderByDescending(tuple => tuple.value).ToArray();

    if (ordered[0].index == label)
    {
        score++;
    }

    Console.WriteLine($"Label: {label}, Predicted: {ordered[0].index}[{ordered[0].value}], Second predicted: {ordered[1].index}[{ordered[1].value}]");
}

double accuracy = (double)score / imagesTest.Length;
Console.WriteLine($"Neural Network Performance: {accuracy * 100}%");
Console.WriteLine("Number of training images: " + imagesTrain.Length);
Console.WriteLine("Number of test images: " + imagesTest.Length);
Console.WriteLine("Number of epochs: " + epochs);


Console.WriteLine("Hello, World!");
