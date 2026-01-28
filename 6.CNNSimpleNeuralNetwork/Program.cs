using GnuPlotWrapper;
using SimpleNeuralNetworkTorchSharp;
using TorchSharp;

var isCudaAvailable = torch.cuda.is_available();
Console.WriteLine("Is CUDA available: " + isCudaAvailable);


// var pathData = "../../../../data/mnist_train_100.csv";
var pathData = "/home/john/Downloads/mnist_train.csv/mnist_train.csv";
var lines = await File.ReadAllLinesAsync(pathData);
var images = lines.Select(line => new MnistImage(line))
    .ToArray();
    // .Take(1000).ToArray();

// Split into training and test sets
var numberOfTestImages = images.Length / 10;
var numberOfTrainImages = images.Length - numberOfTestImages;
var imagesTrain = images.Take(numberOfTrainImages).ToArray();
var imagesTest = images.Skip(numberOfTrainImages).ToArray();

// var grayScaleMatrixImage = GnuPlotHelpers
//     .ShowImage(imagesTrain.First().Pixels, "First Training Image");

// using var gnuPlotWrapper1 = new GnuPlot();
// gnuPlotWrapper1.Start();
// await gnuPlotWrapper1.ExecuteAsync(grayScaleMatrixImage.AsMemory());


const int epochs = 3;
var classifier = new Classifier();
var trainDataset = new MnistDataSet(imagesTrain);

int counter = 0;
for (int epoch = 0; epoch < epochs; epoch++)
{
    Console.WriteLine($"Epoch {epoch + 1}/{epochs}");
    counter = 0;

    foreach ((int label, torch.Tensor? imageValues, torch.Tensor? target) in trainDataset)
    {
        counter++;

        if (counter % 1000 == 0)
        {
            Console.WriteLine($"  Training image {counter}/{trainDataset.Count}");
        }

        using torch.Tensor targetReshaped = target.reshape(1, -1);
        using torch.Tensor imageValuesReshaped = imageValues.reshape(
            // Batch size(one image in shape), channels(greyscale), height, width
            1, 1, 28, 28);
        classifier.Train(imageValuesReshaped, targetReshaped);
    }
}

var trainingLosses = classifier.GetTrainingLosses();

using var gnuPlotWrapper = new GnuPlot();
gnuPlotWrapper.Start();
await gnuPlotWrapper.ExecuteAsync(GnuPlotHelpers.TrainingLosses(trainingLosses).AsMemory());


int score = 0;
var testDataset = new MnistDataSet(imagesTest);
foreach ((int label, torch.Tensor? imageValues, torch.Tensor? target) in testDataset)
{
    using var imageValuesReshaped = imageValues.reshape(
        // Batch size(one image in shape), channels(greyscale), height, width
        1, 1, 28, 28);
    torch.Tensor output = classifier
        .forward(input: imageValuesReshaped)
        .detach();

    (float value, int index)[] ordered = [.. output.data<float>().Select((value, index) => (value, index)).OrderByDescending(tuple => tuple.value)];

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

// Label: 6, Predicted: 6[0.9955011], Second predicted: 4[0.00067303877]
// Label: 0, Predicted: 0[0.9972119], Second predicted: 2[0.0017542661]
// Label: 7, Predicted: 7[0.9999666], Second predicted: 8[0.00061977905]
// Label: 8, Predicted: 8[0.99999595], Second predicted: 9[0.0011516035]
// Label: 9, Predicted: 9[0.9933734], Second predicted: 7[0.0031509795]
// Label: 2, Predicted: 2[0.999483], Second predicted: 8[0.23067267]
// Label: 9, Predicted: 9[0.98020875], Second predicted: 7[0.01554027]
// Label: 5, Predicted: 5[0.9999161], Second predicted: 8[0.0006054291]
// Label: 1, Predicted: 1[0.99913543], Second predicted: 7[0.0049620615]
// Label: 8, Predicted: 8[0.9999995], Second predicted: 3[0.0006777902]
// Label: 3, Predicted: 3[0.99992824], Second predicted: 8[0.0070069963]
// Label: 5, Predicted: 5[0.99940765], Second predicted: 8[0.005628047]
// Label: 6, Predicted: 6[0.99973315], Second predicted: 5[0.00039780219]
// Label: 8, Predicted: 8[0.9896432], Second predicted: 9[0.0003862773]
// Neural Network Performance: 98.61666666666666%
// Number of training images: 54000
// Number of test images: 6000
// Number of epochs: 3
