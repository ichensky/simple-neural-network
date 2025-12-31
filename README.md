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

### Refs: 
- Make Your Own Neural Network: Rashid, Tariq
