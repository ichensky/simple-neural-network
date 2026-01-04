using GnuPlotWrapper;
using SimpleNeuralNetworkTorchSharp;
using TorchSharp;

// var pathData = "../../../../data/mnist_train_100.csv";
var pathData = "/home/john/Downloads/mnist_train.csv/mnist_train.csv";
var lines = await File.ReadAllLinesAsync(pathData);
var images = lines.Select(line => new MnistImage(line))
    .ToArray();
    // .Take(000).ToArray();

// Split into training and test sets
var numberOfTestImages = 1_000;
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


// Debug output:

// Epoch 1/3
// Epoch 2/3
// Epoch 3/3
// Label: 6, Predicted: 6[0.9234286], Second predicted: 1[0.10731403]
// Label: 7, Predicted: 7[0.80352944], Second predicted: 4[0.09078009]
// Label: 4, Predicted: 4[0.9545391], Second predicted: 9[0.17752466]
// Label: 6, Predicted: 6[0.8726566], Second predicted: 1[0.069927305]
// Label: 8, Predicted: 8[0.65236336], Second predicted: 1[0.08555014]
// Label: 0, Predicted: 0[0.9159549], Second predicted: 6[0.032799874]
// Label: 7, Predicted: 7[0.70220625], Second predicted: 9[0.37765577]
// Label: 8, Predicted: 8[0.9184867], Second predicted: 5[0.069465496]
// Label: 3, Predicted: 3[0.9165112], Second predicted: 4[0.031831652]
// Label: 1, Predicted: 1[0.93227255], Second predicted: 4[0.08585575]
// Neural Network Performance: 100%
// Number of training images: 90
// Number of test images: 10
// Number of epochs: 3
