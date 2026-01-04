# Simple neural network in C# 

Simple neural network is the construction of a [Multilayer Perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron) (MLP), which is a classic type of Feedforward Artificial Neural Network.

The architecture is consists of three layers:

`Input Layer`: Receives the raw data (pixels from handwritten digits).

`Hidden Layer`: Processes the signals with weights and an activation function.

`Output Layer`: Produces the final classification (e.g., identifying which digit 0–9 is shown).

#### Key Characteristics of the Network:
`Fully Connected`: Every node in one layer is connected to every node in the next layer.

`Activation Function`: Used the Sigmoid function to introduce non-linearity into the model.

`Learning Mechanism`: It utilizes Backpropagation to calculate errors and Gradient Descent to update the weights between nodes.




A simple neural network has one `hidden layer`. 
In this example, it is trained on MNIST images to be able to recognize hand-written digits.

MNIST image stored in csv
[7, 0, 0,...., 198, 255,...0,]

where 7 - represents image stored in the csv line
    0, 198, 255, ... - grayscale values from 0 to 255

https://pjreddie.com/projects/mnist-in-csv/

Number of training images: 90
Number of test images: 10
Number of epochs: 5
Performance: 100%
```csharp
    // Debug output:

    // Training on MNIST dataset...
    // Label: 6, Predicted: 6[0.9565385366964735], Second predicted: 0[0.08471559056467289]
    // Label: 7, Predicted: 7[0.7782230846342505], Second predicted: 4[0.10997378352093976]
    // Label: 4, Predicted: 4[0.9590104713170882], Second predicted: 9[0.08107556885578783]
    // Label: 6, Predicted: 6[0.9195931776377313], Second predicted: 1[0.06751671433028496]
    // Label: 8, Predicted: 8[0.6958355798493969], Second predicted: 9[0.10310759896822407]
    // Label: 0, Predicted: 0[0.9131043963800642], Second predicted: 6[0.06941072092591712]
    // Label: 7, Predicted: 7[0.7323817320259772], Second predicted: 9[0.6356787895520682]
    // Label: 8, Predicted: 8[0.9521698144542837], Second predicted: 9[0.0916935525540241]
    // Label: 3, Predicted: 3[0.929344564561091], Second predicted: 9[0.07426378958115883]
    // Label: 1, Predicted: 1[0.8501468932454387], Second predicted: 4[0.08364124013770946]
    // Neural Network Performance: 100%
    // Number of training images: 90
    // Number of test images: 10
    // Number of epochs: 5
```

Number of training images: 59000
Number of test images: 1000
Number of epochs: 3
Performance: 97.4%
```csharp
// Label: 8, Predicted: 8[0.99999976], Second predicted: 2[4.4215118E-05]
// Label: 9, Predicted: 9[0.9998324], Second predicted: 7[0.00010894189]
// Label: 2, Predicted: 2[0.9999881], Second predicted: 5[0.0001921707]
// Label: 9, Predicted: 9[0.9991196], Second predicted: 7[0.00014359904]
// Label: 5, Predicted: 5[1], Second predicted: 9[3.5807866E-09]
// Label: 1, Predicted: 1[0.9965064], Second predicted: 7[7.580602E-05]
// Label: 8, Predicted: 8[0.999998], Second predicted: 1[3.4516643E-06]
// Label: 3, Predicted: 3[0.99999917], Second predicted: 8[0.00014642277]
// Label: 5, Predicted: 5[1], Second predicted: 3[2.5686325E-07]
// Label: 6, Predicted: 6[0.99985874], Second predicted: 5[0.00022567397]
// Label: 8, Predicted: 8[0.9999994], Second predicted: 5[1.3962545E-05]
// Neural Network Performance: 97.39999999999999%
// Number of training images: 59000
// Number of test images: 1000
// Number of epochs: 3
```


### Visualization of how neural-network see digits
After training the neural network on 90 images, it stores information about them as images:

![how neural network see digits](/doc/how-neural-network-see-digits.png)

### Training loss 


Number of training images: 90
Number of test images: 10
Number of epochs: 3
![training loss](/doc/training-loss.png)


Number of training images: 59000
Number of test images: 1000
Number of epochs: 3
![training-loss-3epochs-59000-images](/doc/training-loss-3epochs-59000-images.png)


### Projects
`SimpleNeuralNetwork` - impelementation of the MLP without third-party libraries
`SimpleNeuralNetworkTorchSharp` - impelementation of the MLP with `TorchSharp` library

### Refs: 
- Make Your Own Neural Network: Rashid, Tariq
