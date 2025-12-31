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


    const int epochs = 2;
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
    // Label: 6, Predicted: 6[0.9555110138579607], Second predicted: 0[0.11422946053018113]
    // Label: 7, Predicted: 7[0.6107789717181917], Second predicted: 9[0.1895928206857955]
    // Label: 4, Predicted: 4[0.9326265652089964], Second predicted: 6[0.0944038564612339]
    // Label: 6, Predicted: 6[0.9228503677473069], Second predicted: 1[0.05494633682019927]
    // Label: 8, Predicted: 8[0.6673127070837434], Second predicted: 9[0.13679885627027302]
    // Label: 0, Predicted: 0[0.8602105693016026], Second predicted: 6[0.10442636093935491]
    // Label: 7, Predicted: 9[0.6269751657233196], Second predicted: 7[0.5212717658121823]
    // Label: 8, Predicted: 8[0.8776229678775155], Second predicted: 9[0.14153030418956095]
    // Label: 3, Predicted: 3[0.8819434444588291], Second predicted: 1[0.06707529002523892]
    // Label: 1, Predicted: 1[0.8303102184641233], Second predicted: 2[0.07459432168779988]
    // Neural Network Performance: 90%
    // Number of training images: 90
    // Number of test images: 10
    // Number of epochs: 2
}

Console.WriteLine("Hello, World!");
