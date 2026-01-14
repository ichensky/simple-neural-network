using GnuPlotWrapper;
using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;


var realLabels = torch.FloatTensor(new float[] { 1.0f });
var fakeLabels = torch.FloatTensor(new float[] { 0.0f });
var generatorInput = torch.FloatTensor(new float[] { 0.5f });

var discriminator = new Discriminator();
var generator = new Generator();
IList<IList<float>> generatorOutput = [];

for(int i =0;i< 10_000;i++)
{
    // Train Discriminator on real data
    Tensor realData = Discriminator.GenerateReal();
    var lossReal = discriminator.Train(realData, realLabels);

    // Train Discriminator on fake data
    var fakeData = generator.forward(generatorInput).detach();
    var lossFake = discriminator.Train(fakeData, fakeLabels);

    // Train Generator
    generator.Train(discriminator, generatorInput, realLabels);

    if(i % 1000 == 0)
    {
        var output = generator.forward(generatorInput).detach();
        generatorOutput.Add([.. output.data<float>()]);
    }
}


// Plot Discriminator training losses
var discriminatorTrainingLosses = discriminator.GetTrainingLoss();
using var discriminatorGnuPlotWrapper = new GnuPlot();
discriminatorGnuPlotWrapper.Start();
await discriminatorGnuPlotWrapper.ExecuteAsync(GnuPlotHelpers
    .TrainingLosses(discriminatorTrainingLosses, "Discriminator Training Loss").AsMemory());

// Plot Generator training losses
var generatorTrainingLosses = generator.GetTrainingLoss();
using var generatorGnuPlotWrapper = new GnuPlot();
generatorGnuPlotWrapper.Start();
await generatorGnuPlotWrapper.ExecuteAsync(GnuPlotHelpers
    .TrainingLosses(generatorTrainingLosses, "Generator Training Loss").AsMemory());

// Plot Generator outputs 
using var generatorOutputGnuPlotWrapper = new GnuPlot();
generatorOutputGnuPlotWrapper.Start();
await generatorOutputGnuPlotWrapper.ExecuteAsync(GnuPlotHelpers
    .GeneratorOutput(generatorOutput, "Generator Output").AsMemory());


// check correctness
var outputReal = discriminator.forward(Discriminator.GenerateReal());
Console.WriteLine($"Real Data Prediction: {outputReal.item<float>()}");

var outputFake = discriminator.forward(Discriminator.GenerateRandom(4));
Console.WriteLine($"Fake Data Prediction: {outputFake.item<float>()}");


Console.WriteLine("Exiting...");



// def generate_real():
// real_data = torch.FloatTensor([1, 0, 1, 0])
// return real_data



public class Generator : Module<Tensor, Tensor>
{
    private readonly Sequential model;
    private readonly MSELoss lossFunction;
    private readonly SGD optimizer;
    private List<float> trainingLoss = [];

    public Generator() :base("Generator")
    {
        model = nn.Sequential(
            nn.Linear(1, 3),
            nn.Sigmoid(),
            nn.Linear(3, 4),
            nn.Sigmoid()
        );

        // stochastic gradient descent
        optimizer = optim.SGD(model.parameters(), 0.01f);
    }


    public static Tensor GenerateReal()
    {
        Random random = new();

        // Tensor realData = torch.FloatTensor(new float[] { 1, 0, 1, 0 });
        Tensor realData = torch.FloatTensor(new float[] { random.NextSingle()*0.2f + 0.8f,
                                                      random.NextSingle()*0.2f,
                                                      random.NextSingle()*0.2f + 0.8f,
                                                      random.NextSingle()*0.2f });

        return realData;
    }

    public static Tensor GenerateRandom(int size)
    {
        Random random = new();

        Tensor randomData = torch.rand(size);
        return randomData;
    }

    public Tensor Train(Discriminator discriminator, Tensor input, Tensor target)
    {
        Tensor geneartorOutput = model.forward(input);
        Tensor discriminatorOutput = discriminator.forward(geneartorOutput);
        Tensor loss = discriminator.LossFunction.forward(discriminatorOutput, target);

        trainingLoss.Add(loss.item<float>());

        optimizer.zero_grad();

        loss.backward();
        optimizer.step();

        return loss;
    }

    public IList<float> GetTrainingLoss() => trainingLoss;

    public override Tensor forward(Tensor input)
    {
        return model.forward(input);
    }
}

public class Discriminator : Module<Tensor, Tensor>
{
    private readonly Sequential model;
    private readonly MSELoss lossFunction;
    private readonly SGD optimizer;
    private List<float> trainingLoss = [];

    public Discriminator() :base("Discriminator")
    {
        model = nn.Sequential(
            nn.Linear(4, 3),
            nn.Sigmoid(),
            nn.Linear(3, 1),
            nn.Sigmoid()
        );

        // mean squared error loss
        lossFunction = nn.MSELoss();

        // stochastic gradient descent
        optimizer = optim.SGD(model.parameters(), 0.01f);
    }

    public MSELoss LossFunction => lossFunction;

    public static Tensor GenerateReal()
    {
        Random random = new();

        // Tensor realData = torch.FloatTensor(new float[] { 1, 0, 1, 0 });
        Tensor realData = torch.FloatTensor(new float[] { random.NextSingle()*0.2f + 0.8f,
                                                      random.NextSingle()*0.2f,
                                                      random.NextSingle()*0.2f + 0.8f,
                                                      random.NextSingle()*0.2f });

        return realData;
    }

    public static Tensor GenerateRandom(int size)
    {
        Random random = new();

        Tensor randomData = torch.rand(size);
        return randomData;
    }

    public Tensor Train(Tensor input, Tensor target)
    {
        Tensor output = model.forward(input);
        Tensor loss = lossFunction.forward(output, target);

        trainingLoss.Add(loss.item<float>());

        optimizer.zero_grad();

        loss.backward();
        optimizer.step();

        return loss;
    }

    public IList<float> GetTrainingLoss() => trainingLoss;

    public override Tensor forward(Tensor input)
    {
        return model.forward(input);
    }
}


