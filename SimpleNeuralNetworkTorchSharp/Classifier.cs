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
            torch.nn.Linear(784, 200),
            torch.nn.LeakyReLU(0.02),

            torch.nn.LayerNorm(200),

            torch.nn.Linear(200, 10),
            torch.nn.Sigmoid()
        );

        // MSELoss - Mean Squared Error Loss - good for regression
        // loss_function = torch.nn.MSELoss();
        // BCELoss - Binary Cross Entropy Loss - good for classification
        loss_function = torch.nn.BCELoss();

        // create optimiser, using simple stochastic gradient descent
        // optimiser = torch.optim.SGD(model.parameters(), 0.01);
        optimiser = torch.optim.Adam(model.parameters());
    }

    public override torch.Tensor forward(torch.Tensor input)
    {
        return model.forward(input);
    }

    public void Train(torch.Tensor input, torch.Tensor target)
    {
        // Forward pass: compute the model output
        torch.Tensor output = forward(input);

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