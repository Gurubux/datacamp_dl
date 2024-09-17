# datacamp_dl
skill track  Deep Learning in Python   

# Chapter 1: Introduction to PyTorch, a Deep Learning Library

1. Introduction to deep learning with PyTorch  
```
import torch  
torch.tensor([[],[]])  
torch.from_numpy(np.array(array))
```

2. 2-layer network
```
import touch.nn as nn  
input_tensor = torch.tensor([[0.3471, 0.4547, -0.2356]])  
linear_layer = nn.Linear(in_features=3, out_features=2)  
output = linear_layer(input_tensor)
print(output)

linear_layer.weight
linear_layer.bias
```
```
# Create network with three linear layers
model = nn.Sequential(nn.Linear(10, 18),nn.Linear(18, 20), nn.Linear(20, 5))
```
**Activation function**  
Activation function as the last Layer      
Binary-Sigmoid classification    
Multiclass-Sofmax classification  

# Chapter 2: Training Our First Neural Network with PyTorch

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

**In regression,**  
```
model = nn.Sequential(nn.Linear(6,4), nn.Linear(4,1))  
output = model(input_data)
```
Basically, no Activation function as the last layer makes in the Regression model. 

**Multiclass**  
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


**Loss Function**  
- Gives Feedback to model during training
- Takes in y and  yhat and returns outputs a float.

loss = F(y, yhat)  
Loss Function  
1. **CrossEntropyLoss** == Classification Problem  
2. **MSELoss** == Regression Problem
**one_hot**  
```
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

y = [2]
scores = torch.tensor([[0.1, 6.0, -2.0, 3.2]])

# Create a one-hot encoded vector of the label y
one_hot_label = F.one_hot(torch.tensor(y), scores.shape[1])

# Create the cross entropy loss function
criterion = CrossEntropyLoss()

# Calculate the cross entropy loss
loss = criterion(scores.double(), one_hot_label.double())

print(loss)
```

**Using Derivatives**  
Gradient == Derivatives

![image](https://github.com/user-attachments/assets/0b426a31-4fe6-40c6-a265-4f15371013b6)  

Convex Functions: Only one local minimum
Non-Convex Functions: Two or more local minimum
![image](https://github.com/user-attachments/assets/7b3095e6-1c98-492f-9bb8-438628412c9a)

**The goal is to find the Global Minimum of non-Complex Function**
- Using Gradient Descent
- In pytorch, an optimizer takes care of weight updates
- Most common: SGD (Stochastic GD)
```
import torch.optim as optim # Create the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)
# Optimizer handles updating model parameters (or weights) after calculation of localgradients
optimizer.step()

```
### Update the weight of a network  
**Manual**
```
model = nn.Sequential(nn.Linear(16, 8),
                      nn.Sigmoid(),
                      nn.Linear(8, 2))

# Access the weight of the first linear layer
weight_0 = model[0].weight

# Access the bias of the second linear layer
bias_1 = model[2].bias

weight0 = model[0].weight
weight1 = model[1].weight
weight2 = model[2].weight

# Access the gradients of the weight of each linear layer
grads0 = weight0.grad
grads1 = weight1.grad
grads2 = weight2.grad

# Update the weights using the learning rate and the gradients
weight0 =  weight0.data - lr * grads0
weight1 =  weight1.data - lr * grads1
weight2 =  weight2.data - lr * grads2
```
**Using the PyTorch optimizer**
```
# Create the optimizer
optimizer = optim.SGD(model.parameters(), lr=0.001)

loss = criterion(pred, target)
loss.backward()

# Update the model's parameters using the optimizer
optimizer.step()
```

## Training a neural network

1. Create a model
2. Choose a loss function
3. Create a dataset
4. Define an optimizer
5. Run a training loop, where for each sample of the dataset, we repeat:
   - Calculating loss (forward pass)
   - Calculating local gradients
   - Updating model parameters


Steps 1 to 4
```
# Create the dataset and the dataloader
dataset = TensorDataset(torch.tensor(features).float(), torch.tensor(target).float())
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Create the model
model = nn.Sequential(nn.Linear(4, 2),
                      nn.Linear(2, 1))

# Create the loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)
```

Step 5
```
# Loop through the dataset multiple times
for epoch in range(num_epochs):
    for data in dataloader:
        
        # Set the gradients to zero
        optimizer.zero_grad()
        
        # Get feature and target from the data loader
        feature, target = data
        
        # Run a forward pass
        pred = model(feature)
        
        # Compute loss and gradients
        loss = criterion(pred, target)
        loss.backward()
        
        # Update the parameters
        optimizer.step()
```
1. Loop through epochs: The outer loop iterates over a specified number of training cycles (epochs).
2. **Loop through data**: For each epoch, the inner loop iterates through batches of data provided by the dataloader.
3. **Reset gradients**: optimizer.zero_grad() resets the gradients to prevent accumulation from previous batches.
4. **Forward pass**: The model makes predictions (pred = model(feature)) based on the current features.
5. **Loss computation**: The loss is calculated based on the model's predictions and the true target values using the specified criterion.
6. **Backward pass**: loss.backward() computes the gradients of the loss with respect to the model's parameters.
7. **Update weights**: optimizer.step() updates the model's parameters based on the gradients and the learning rate.


```
y_hat = np.array(10)
y = np.array(1)

# Calculate the MSELoss using NumPy
mse_numpy = np.mean((y_hat - y)**2)

# Create the MSELoss function
criterion = nn.MSELoss()

# Calculate the MSELoss using the created loss function
mse_pytorch = criterion(torch.tensor(y_hat).float(), torch.tensor(y).float())
print(mse_pytorch)
```

```
# Loop over the number of epochs and the dataloader
for i in range(num_epochs):
  for data in dataloader:
    # Set the gradients to zero
    optimizer.zero_grad()
    # Run a forward pass
    feature, target = data
    prediction = model(feature)    
    # Calculate the loss
    loss = criterion(prediction, target)    
    # Compute the gradients
    loss.backward()
    # Update the model's parameters
    optimizer.step()
show_results(model, dataloader)
```

# Chapter 3: Neural Network Architecture and Hyperparameters  
## Discovering activation functions between layers
Sigmoid has a Vanishing Gradient Problem during backpropagation
- Approaches 0 for low and high values of x
- Cause the function to Saturate

1. ReLU
So, We need ReLU- Rectified Linear Unit
  - f(x) = max(x,0)
  - Positive x, f(x) is x
  - Negative x, f(x) is 0
  - Thus, Overcoming vanishing gradient problem
  ```
  #Pytorch
  relu = nn.Relu()
  ```
2. Leaky ReLU
  - Same for positive x
  - For negative x, it multiplies the x with a small coefficient (defaulted to 0.01)
  - Thus, The gradients for negative x are never NULL
  ```
  leaky_relu = nn.LeakyReLU(negative_slope = 0.05)
  ```
That's correct! Leaky ReLU is another very popular activation function found in modern architecture. By never setting the gradients to zero, it allows every parameter of the model to keep learning.    

### Implementing ReLU

### Implementing leaky ReLU

### Understanding activation functions

## A deeper dive into neural network architecture

### Counting the number of parameters

### Manipulating the capacity of a network

## Learning rate and momentum

### Experimenting with learning rate

### Experimenting with momentum

## Layer initialization and transfer learning

### Fine-tuning process

### Freeze layers of a model

### Layer initialization




# CHpater 4: Evaluating and Improving Models
