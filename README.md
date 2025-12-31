# Simple neural network in C# 

A simple neural network has one `hidden layer`. 
In this example, it is trained on MNIST images to be able to recognize hand-written digits.

MNIST image stored in csv
[7, 0, 0,...., 198, 255,...0,]

where 7 - represents image stored in the csv line

https://pjreddie.com/projects/mnist-in-csv/

```csharp
    // Debug output:

    // Training on MNIST dataset...
    // Label: 6, Predicted: 6[0.9753125867541408], Second predicted: 4[0.12607054094009107]
    // Label: 7, Predicted: 4[0.4068498051873584], Second predicted: 7[0.39002865329624276]
    // Label: 4, Predicted: 4[0.9376553441117896], Second predicted: 6[0.1813234321882868]
    // Label: 6, Predicted: 6[0.9758362393808847], Second predicted: 4[0.04960161526202514]
    // Label: 8, Predicted: 8[0.48849705799545357], Second predicted: 1[0.16744527168546258]
    // Label: 0, Predicted: 0[0.7661083439674404], Second predicted: 6[0.13836600786077213]
    // Label: 7, Predicted: 9[0.4626150821354045], Second predicted: 7[0.33149310403425697]
    // Label: 8, Predicted: 8[0.697151813098993], Second predicted: 1[0.07297951557083256]
    // Label: 3, Predicted: 3[0.8479082152724998], Second predicted: 1[0.09340656902348322]
    // Label: 1, Predicted: 1[0.8059728365313362], Second predicted: 8[0.18865839000648169]
    // Neural Network Performance: 80%
```

### Refs: 
- Make Your Own Neural Network: Rashid, Tariq
