using TorchSharp;
using TorchSharp.Modules;

namespace SimpleNeuralNetworkTorchSharp;

public class Classifier: torch.nn.Module<torch.Tensor, torch.Tensor>
{
    private readonly Sequential model;
    private readonly BCELoss loss_function;
    private readonly Adam optimiser;

    private IList<double> trainingLosses = [];

    public Classifier(): base("Classifier")
    {
        model = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels: 1, out_channels: 10, kernel_size: 5, stride: 2),
            torch.nn.LeakyReLU(0.02),
            torch.nn.BatchNorm2d(num_features: 10),

            torch.nn.Conv2d(in_channels: 10, out_channels: 10, kernel_size: 3, stride: 2),
            torch.nn.LeakyReLU(0.02),
            torch.nn.BatchNorm2d(num_features: 10),

            torch.nn.Flatten(),
            torch.nn.Linear(250, 10),
            torch.nn.Sigmoid()
        );

        // Register the components of the model so that their parameters are included in this module's parameters
        RegisterComponents();

        // MSELoss - Mean Squared Error Loss - good for regression
        // loss_function = torch.nn.MSELoss();
        // BCELoss - Binary Cross Entropy Loss - good for classification
        loss_function = torch.nn.BCELoss();

        // create optimiser, using simple stochastic gradient descent
        // optimiser = torch.optim.SGD(model.parameters(), 0.01);
        optimiser = torch.optim.Adam(this.parameters());
    }

    public override torch.Tensor forward(torch.Tensor input) => model.forward(input);

    public void Train(torch.Tensor input, torch.Tensor target)
    {
        // Forward pass: compute the model output
        torch.Tensor output = forward(input);
        output.ToString(TensorStringStyle.Numpy);


        // Compute the loss between the model output and the target
        torch.Tensor loss = loss_function.forward(output, target);
        
        // The gradients are set to zero before starting to do backpropragation
        optimiser.zero_grad();

        // Gradients are computed working backwards from the loss function
        loss.backward();

        // The optimiser updates the model parameters based on the computed gradients
        optimiser.step();

        trainingLosses.Add(loss.ToSingle());
    }

    public IList<double> GetTrainingLosses() => trainingLosses;
}