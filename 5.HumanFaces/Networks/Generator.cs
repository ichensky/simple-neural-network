using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace HumanFaces.Networks;

public class Generator : Module<Tensor, Tensor>
{
    private readonly Sequential model;
    private readonly Adam optimizer;

    public Generator() :base("Generator")
    {
        var imageSize = 178 * 218 * 3;
        model = nn.Sequential(
            nn.Linear(100, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, imageSize),
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


