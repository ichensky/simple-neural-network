using GnuPlotWrapper;
using SimpleNeuralNetwork;
using SimpleNeuralNetwork.Helpers;

// Simple test of the SimpleNeuralNetwork
{
    Random rand = new(123);

    const double learningRate = 0.3;
    const int inputNodesCount = 4;
    const int hiddenNodesCount = 3;
    const int outputNodesCount = 3;

    Vector inputs = new([1.0, 0.5, -1.5, 2.0]);
    Vector targets = new([0.5, 0.1, 0.9]);

    // Initialize the neural network with random weights
    var neuralNetwork = new SimpleNeuralNetwork.SimpleNeuralNetwork(rand, inputNodesCount, hiddenNodesCount, outputNodesCount, learningRate);
    neuralNetwork.InitializeWeights();

    // Query the neural network output vector before training
    var query = neuralNetwork.Query(inputs);

    var dotSim = query.DotProduct(targets);
    var cosSim = query.CosineProduct(targets);

    Console.WriteLine("Initial query output: " + string.Join(", ", query.Data));
    Console.WriteLine($"Similarity to target before training: Dot: {dotSim} ; Cosine: {cosSim}");

    // Train the neural network with the input and target vectors
    neuralNetwork.Train( inputs, targets);

    // Query the neural network output vector after training
    var queryAfterTraining = neuralNetwork.Query(inputs);

    var dotSimAfter = queryAfterTraining.DotProduct(targets);
    var cosSimAfter = queryAfterTraining.CosineProduct(targets);

    Console.WriteLine("Query output after training: " + string.Join(", ", queryAfterTraining.Data));
    Console.WriteLine($"Similarity to target after training: Dot: {dotSimAfter} ; Cosine: {cosSimAfter}");

    Console.WriteLine($"Similarity improvement: Dot: {dotSimAfter - dotSim} ; Cosine: {cosSimAfter - cosSim}");

    // Debug output:

    // Initial query output: 0.5474683187393832, 0.47559067819266515, 0.562841997832462
    // Similarity to target before training: Dot: 0.8278510252381739 ; Cosine: 0.8718143629007491
    // Query output after training: 0.5508283176286549, 0.4807251931908334, 0.5730076558713788
    // Similarity to target after training: Dot: 0.8391935684176518 ; Cosine: 0.8733805824495947
    // Similarity improvement: Dot: 0.011342543179477915 ; Cosine: 0.0015662195488456154
}

Console.WriteLine("-----------------------------------");

// Simple test of the SimpleNeuralNetwork on MNIST dataset
{
    Console.WriteLine("Training on MNIST dataset...");
    var pathData = "data/mnist_train_100.csv";
    var lines = await File.ReadAllLinesAsync(pathData);
    var images = lines.Select(line => new MnistImage(line));

    // Split into training and test sets
    var numberOfTestImages = 10;
    var numberOfTrainImages = lines.Length - numberOfTestImages;
    var imagesTrain = images.Take(numberOfTrainImages).ToArray();
    var imagesTest = images.Skip(numberOfTrainImages).ToArray();


    Random rand = new(123);

    var num = rand.NextDouble();


    const int epochs = 5;
    const double learningRate = 0.2;
    const int inputNodesCount = 784;
    const int hiddenNodesCount = 100;
    const int outputNodesCount = 10;

    var neuralNetwork = new SimpleNeuralNetwork.SimpleNeuralNetwork(rand, inputNodesCount, hiddenNodesCount, outputNodesCount, learningRate);
    neuralNetwork.InitializeWeights();

    for (int e = 0; e < epochs; e++)
    {
        foreach (var image in imagesTrain)
        {
            Vector inputLayer = new([.. image.Pixels.Select(pixel => (pixel / 255d) * 0.99 + 0.01)]);

            Vector target = new([.. Enumerable.Repeat(0.01, 10)]);
            target.Data[image.Label] = 0.99;

            // Train the neural network with the input and target vectors
            neuralNetwork.Train(inputLayer, target);
        }
    }

    int score = 0;
    foreach (var image in imagesTest)
    {
        Vector inputLayer = new([.. image.Pixels.Select(pixel => (pixel / 255d) * 0.99 + 0.01)]);

        // Query the neural network output vector
        Vector result = neuralNetwork.Query(inputLayer);
        var ordered = result.Data.Select((value, index) => (value, index)).OrderByDescending(tuple => tuple.value).ToArray();
        if (ordered[0].index == image.Label)
        {
            score++;
        }

        Console.WriteLine($"Label: {image.Label}, Predicted: {ordered[0].index}[{ordered[0].value}], Second predicted: {ordered[1].index}[{ordered[1].value}]");
    }
    Console.WriteLine($"Neural Network Performance: {(double)score / imagesTest.Length * 100}%");
    Console.WriteLine("Number of training images: " + imagesTrain.Length);
    Console.WriteLine("Number of test images: " + imagesTest.Length);
    Console.WriteLine("Number of epochs: " + epochs);


    // Debug output:

    // Training on MNIST dataset...
    // Label: 6, Predicted: 6[0.9565385366964735], Second predicted: 0[0.08471559056467289]
    // Label: 7, Predicted: 7[0.7782230846342505], Second predicted: 4[0.10997378352093976]
    // Label: 4, Predicted: 4[0.9590104713170882], Second predicted: 9[0.08107556885578783]
    // Label: 6, Predicted: 6[0.9195931776377313], Second predicted: 1[0.06751671433028496]
    // Label: 8, Predicted: 8[0.6958355798493969], Second predicted: 9[0.10310759896822407]
    // Label: 0, Predicted: 0[0.9131043963800642], Second predicted: 6[0.06941072092591712]
    // Label: 7, Predicted: 7[0.7323817320259772], Second predicted: 9[0.6356787895520682]
    // Label: 8, Predicted: 8[0.9521698144542837], Second predicted: 9[0.0916935525540241]
    // Label: 3, Predicted: 3[0.929344564561091], Second predicted: 9[0.07426378958115883]
    // Label: 1, Predicted: 1[0.8501468932454387], Second predicted: 4[0.08364124013770946]
    // Neural Network Performance: 100%
    // Number of training images: 90
    // Number of test images: 10
    // Number of epochs: 5


    // Generate backquery images(to visualize what the neural network "thinks" about each digit)
    {
        var imagesDataLines = new List<string>();

        for (int i = 0; i < 10; i++)
        {
            Vector target = new([.. Enumerable.Repeat(0.01, 10)]);
            target.Data[i] = 0.99;

            var image = neuralNetwork.BackQuery(target);

            var gnuPlot = new GnuPlot();
            gnuPlot.Start();
            await gnuPlot.ExecuteAsync(GnuPlotHelpers.ConvertToGrayScaleMatrixImageScript(image.Data).AsMemory());
        }        
    }
}

Console.WriteLine("Hello, World!");
