using TorchSharp;
using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace CNNHumanFaces.Networks;

public class View : Module<Tensor, Tensor>
{
    private readonly long[] shape;
    public View(params long[] shape) : base("View") => this.shape = shape;
    public override Tensor forward(Tensor input) => input.view(shape);
}

public class Generator : Module<Tensor, Tensor>
{
    private readonly Sequential model;
    private readonly Adam optimizer;

    public Generator() :base("Generator")
    {
        model = nn.Sequential(
            nn.Linear(inputSize: 100, outputSize: 3*11*11),
            nn.GELU(),
            // nn.LeakyReLU(0.2),

            new View(1, 3, 11, 11),

            nn.ConvTranspose2d(in_channels: 3, out_channels: 256, kernel_size: 8, stride: 2),
            nn.BatchNorm2d(num_features: 256),
            nn.GELU(),
            // nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels: 256, out_channels: 256, kernel_size: 8, stride: 2),
            nn.BatchNorm2d(num_features: 256),
            nn.GELU(),
            // nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(in_channels: 256, out_channels: 3, kernel_size: 8, stride: 2, padding: 1),
            nn.BatchNorm2d(num_features: 3),

            nn.Sigmoid()
        );

        RegisterComponents();

        // stochastic gradient descent
        // optimizer = optim.SGD(model.parameters(), 0.01f);
        optimizer = optim.Adam(this.parameters(), 0.0001f);
    }

    public Tensor Train(Discriminator discriminator, Tensor input, Tensor target)
    {
        using Tensor geneartorOutput = this.forward(input);
        using Tensor discriminatorOutput = discriminator.forward(geneartorOutput);
        Tensor loss = discriminator.LossFunction.forward(discriminatorOutput, target);

        this.TrainingLoss.Add(loss.item<float>());

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        return loss;
    }

    public IList<float> TrainingLoss { get; } = [];

    public override Tensor forward(Tensor input) => model.forward(input);
}


