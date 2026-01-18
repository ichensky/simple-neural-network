using System.Collections;
using SimpleNeuralNetworkTorchSharp;
using TorchSharp;

namespace SimpleNeuralNetworkTorchSharp;

public class MnistDataSet(MnistImage[] images) 
    : torch.utils.data.Dataset<(int label, torch.Tensor imageValues, torch.Tensor target)>,
    IEnumerable<(int label, torch.Tensor imageValues, torch.Tensor target)>
{
    private readonly MnistImage[] images = images;

    public override long Count => images.Length;

    public IEnumerator<(int label, torch.Tensor imageValues, torch.Tensor target)> GetEnumerator()
    {
        for (long i = 0; i < images.Length; i++)
        {
            yield return GetTensor(i);
        }
    }

    public override (int label, torch.Tensor imageValues, torch.Tensor target) GetTensor(long index)
    {
        int label = images[index].Label;
        torch.Tensor target = torch.zeros([10]);
        target[label] = 1.0f;

        torch.Tensor imageValues = torch.tensor(images[index].Pixels)
            .to_type(torch.ScalarType.Float32)
            .div(255.0f);
        return (label, imageValues, target);
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }
}
