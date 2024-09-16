# datacamp_dl
skill track  Deep Learning in Python   
1. Introduction to deep learning with PyTorch  
import torch  
torch.tensor([[],[]])  
torch.from_numpy(np.array(array))

2. 2-layer network
import touch.nn as nn  
input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356]])  
linear_layer = nn.Linear(in_features=3, out_features=2)  
output = linear_layer(input_tensor)
print(output)

linear_layer.weight
linear_layer.bias


# Create network with three linear layers
model = nn.Sequential(nn.Linear(10, 18),nn.Linear(18, 20), nn.Linear(20, 5))

"Activation function"  
Activation function as the last Layer      
Binary-Sigmoid classification    
Multiclass-Sofmax classification  

**Forward Pass**  
Running a Forward pass through a network  
Predicting using a model
- Binary and Multi-class classification
- Regression
  
**Is there a Backward Pass?**  
Also called backpropagation, it is used to update weights and biases during training.  

**Training Loop**  
1. Propagate data forward
2. Compare outputs to true values (ground truth)
3. Backpropagate to update mode weights and biases
4. Repeat until weights and biases are tuned to produce useful outputs 

In regression,  
model = nn.Sequential(nn.Linear(6,4), nn.Linear(4,1))  
output = model(input_data)  
Basically, no Activation function as the last layer makes in Regression model.   
```
  import torch  
  import torch.nn as nn   
  input_tensor = torch.Tensor([[3, 4, 6, 7, 10, 12, 2, 3, 6, 8, 9]])  
  # Update network below to perform a multi-class classification with four labels  
  model = nn.Sequential(  
    nn.Linear(11, 20),  
    nn.Linear(20, 12),  
    nn.Linear(12, 6),  
    nn.Linear(6, 4),   
    nn.Softmax()  
  )  
  output = model(input_tensor)  
  print(output)  
  ```
