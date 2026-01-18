using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace HandwrittenDigits.Networks;

public class Discriminator : Module<Tensor, Tensor>
{
    private readonly Sequential model;
    private readonly Adam optimizer;

    public Discriminator() :base("Discriminator")
    {
        model = nn.Sequential(
            nn.Linear(784, 200),
            nn.LeakyReLU(0.02),

            nn.LayerNorm(200),

            nn.Linear(200, 1),
            nn.Sigmoid()
        );

        RegisterComponents();

        // mean squared error loss
        LossFunction = nn.BCELoss();

        // stochastic gradient descent
        // optimizer = optim.SGD(model.parameters(), 0.01f);

        optimizer = optim.Adam(this.parameters(), 0.0001f);

    }

    public BCELoss LossFunction { get; }

    public Tensor Train(Tensor input, Tensor target)
    {
        using Tensor output = this.forward(input);
        Tensor loss = LossFunction.forward(output, target);

        this.TrainingLoss.Add(loss.item<float>());

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        return loss;
    }

    public IList<float> TrainingLoss { get; } = [];
    
    public override Tensor forward(Tensor input) => model.forward(input);
}


