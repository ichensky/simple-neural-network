using TorchSharp.Modules;
using static TorchSharp.torch;
using static TorchSharp.torch.nn;

namespace CNNHumanFaces.Networks;

public class Discriminator : Module<Tensor, Tensor>
{
    private readonly Sequential model;
    private readonly Adam optimizer;

    public Discriminator() :base("Discriminator")
    {
        model = nn.Sequential(
            nn.Conv2d(in_channels: 3, out_channels: 256, kernel_size: 8, stride: 2),
            nn.BatchNorm2d(num_features: 256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(in_channels: 256, out_channels: 256, kernel_size: 8, stride: 2),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(num_features: 256),
            
            nn.Conv2d(in_channels: 256, out_channels: 3, kernel_size: 8, stride: 2),
            nn.LeakyReLU(0.2),

            nn.Flatten(),
            nn.Linear(3*10*10, 1),
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


