using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace HandwrittenDigits.Networks;

public class Generator : Module<Tensor, Tensor, Tensor>
{
    private readonly Sequential model;
    private readonly Adam optimizer;

    public Generator() :base("Generator")
    {
        model = nn.Sequential(
            nn.Linear(100 + 10, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 784),
            nn.Sigmoid()
        );

        RegisterComponents();

        // stochastic gradient descent
        // optimizer = optim.SGD(model.parameters(), 0.01f);
        optimizer = optim.Adam(this.parameters(), 0.0001f);
    }

    public Tensor Train(Discriminator discriminator, Tensor input, Tensor labelInput, Tensor target)
    {
        using Tensor geneartorOutput = this.forward(input, labelInput);
        using Tensor discriminatorOutput = discriminator.forward(geneartorOutput, labelInput);
        Tensor loss = discriminator.LossFunction.forward(discriminatorOutput, target);

        this.TrainingLoss.Add(loss.item<float>());

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        return loss;
    }

    public IList<float> TrainingLoss { get; } = [];

    public override Tensor forward(Tensor imageInput, Tensor labelInput)
    {
        return model.forward(cat(new[] { imageInput, labelInput }));
    }
}


