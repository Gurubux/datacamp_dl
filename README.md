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

### Implementing ReLU
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
### Implementing leaky ReLU
2. Leaky ReLU
  - Same for positive x
  - For negative x, it multiplies the x with a small coefficient (defaulted to 0.01)
  - Thus, The gradients for negative x are never NULL
  ```
  leaky_relu = nn.LeakyReLU(negative_slope = 0.05)
  ```
That's correct! Leaky ReLU is another very popular activation function found in modern architecture. By never setting the gradients to zero, it allows every parameter of the model to keep learning.    

A good rule of thumb is to use ReLU as the default activation function in your models (except for the last layer).


### Understanding activation functions
## A deeper dive into neural network architecture
- Full connected Layers
    - input - n_features (fixed)
    - output - n_classes (fixed)
    - hidden - Higher the hidden layers = increasing parameter = increasing model capacity

### Counting the number of parameters
  ![image](https://github.com/user-attachments/assets/16e87720-c4da-49de-9e05-e3d0c6b0d1d0)

### Manipulating the capacity of a network
```
n_features = 8
n_classes = 2

input_tensor = torch.Tensor([[3, 4, 6, 2, 3, 6, 8, 9]])

# Create a neural network with more than 120 parameters
model = nn.Sequential(nn.Linear(n_features, 8),
                      nn.Linear(8, 4),
                      nn.Linear(4, 2),
                      nn.Linear(2, n_classes))

output = model(input_tensor)

print(calculate_capacity(model))
```

## Learning rate and momentum
**Training a neural network = Solving an Optimization Problem**  
 ![image](https://github.com/user-attachments/assets/15a121d8-8417-481b-89fb-e3a9deddfa08)

### Experimenting with learning rate
![image](https://github.com/user-attachments/assets/213c6513-1e6e-414d-9bc9-39a9cc61ec3a)

![image](https://github.com/user-attachments/assets/c1f89070-952a-431f-931c-71847c74937b)

![image](https://github.com/user-attachments/assets/521cc822-98be-4a49-9606-ce5f4ecceddb)

### Experimenting with momentum
Loss function = Non-Convex  
![image](https://github.com/user-attachments/assets/25cd430f-d82a-487d-b25a-7e173b308ded)  

![image](https://github.com/user-attachments/assets/696cf8b7-6bc5-4e53-a56e-e10a89c26c51)

![image](https://github.com/user-attachments/assets/6d56f86b-4ba2-4da6-8a93-b91678bcd2d7)

Momentum and learning rate are critical to the training of your neural network. A good rule of thumb is to start with a learning rate of 0.001 and a momentum of 0.95.


## Layer initialization and transfer learning
### Layer initialization
### Fine-tuning process
### Freeze layers of a model


Here’s a summary of the slides related to **Layer Initialization** and **Transfer Learning/Fine-Tuning**:

### **Layer Initialization (1)**
- **Initial weights**: When a neural network layer is created, its weights are initialized to small random values.
- **Importance of initialization**: If weights are not normalized, they can cause the network outputs to either explode or vanish, which affects learning.
- Example code in PyTorch:
  ```python
  import torch.nn as nn
  layer = nn.Linear(64, 128)
  print(layer.weight.min(), layer.weight.max())
  ```
  - Output: The weights are initialized to small values, for example: 
    ```
    (tensor(-0.1250), tensor(0.1250))
    ```

### **Layer Initialization (2)**
- Weights can be initialized using different methods, such as **uniform distributions** or **normal distributions**.
- Example code for initializing weights using a uniform distribution in PyTorch:
  ```python
  nn.init.uniform_(layer.weight)
  ```
  - After initialization, you can check the new weight range:
    ```python
    print(layer.weight.min(), layer.weight.max())
    ```
  - This can output something like:
    ```
    (tensor(0.0002), tensor(1.0000))
    ```

### **Transfer Learning and Fine Tuning (1)**
- **Transfer learning**: This technique involves using a pre-trained model on a new task to speed up the learning process, especially when the new dataset is smaller or similar to the original dataset.
  - For example, a model trained on a large dataset of US salaries can be reused for a smaller dataset of European salaries.
- Example code to save and load a layer in PyTorch:
  ```python
  import torch
  torch.save(layer, 'layer.pth')  # Save the layer
  new_layer = torch.load('layer.pth')  # Load the saved layer
  ```

### **Transfer Learning and Fine-Tuning**
- **Fine-tuning**: A form of transfer learning where certain layers of the model are "frozen" (i.e., their weights are not updated during training), and only the remaining layers are trained.
- A common approach is to freeze the **early layers** (those closer to the input) and fine-tune the **later layers** (closer to the output).
- Example code for freezing layers in PyTorch:
  ```python
  for name, param in model.named_parameters():
      if name == '0.weight':
          param.requires_grad = False  # Freeze the first layer's weights
  ```

These slides cover techniques to ensure proper weight initialization and how to leverage transfer learning to accelerate model training, along with fine-tuning for better performance on new tasks.  

**Choosing which layer to freeze is an empirical process but a good rule of thumb is to start with the first layers and go deeper.**

# Chpater 4: Evaluating and Improving Models

```python
import numpy as np
import torch
from torch.utils.data import TensorDataset

np_features = np.array(np.random.rand(12, 8))
np_target = np.array(np.random.rand(12, 1))

# Convert arrays to PyTorch tensors
torch_features = torch.tensor(np_features).float()
torch_target = torch.tensor(np_target).float()

# Create a TensorDataset from two tensors
dataset = TensorDataset(torch_features, torch_target)

# Return the last element of this dataset
print(dataset[-1])
```

### From data loading to running a forward pass
```python
# Load the different columns into two PyTorch tensors
features = torch.tensor(dataframe[['ph', 'Sulfate', 'Conductivity', 'Organic_carbon']].to_numpy()).float()
target = torch.tensor(dataframe['Potability'].to_numpy()).float()

# Create a dataset from the two generated tensors
dataset = TensorDataset(features, target)

# Create a dataloader using the above dataset
dataloader = DataLoader(dataset, shuffle=True, batch_size=2)
x, y = next(iter(dataloader))

# Create a model using the nn.Sequential API
model = nn.Sequential(nn.Linear(4, 4),nn.Linear(4, 1))
output = model(features)
print(output)
```

```python
# Set the model to evaluation mode
model.eval()
validation_loss = 0.0

with torch.no_grad():
  
  for data in validationloader:
    
      outputs = model(data[0])
      loss = criterion(outputs, data[1])
      
      # Sum the current loss to the validation_loss variable
      validation_loss += loss.item()
      
# Calculate the mean loss value
validation_loss_epoch = validation_loss / len(validationloader)
print(validation_loss_epoch)

# Set the model back to training mode
model.train()
```

```python
# Create accuracy metric using torch metrics
metric = torchmetrics.Accuracy(task="multiclass", num_classes=3)
for data in dataloader:
    features, labels = data
    outputs = model(features)
    
    # Calculate accuracy over the batch
    acc = metric(outputs.softmax(dim=-1), labels.argmax(dim=-1))
    
# Calculate accuracy over the whole epoch
acc = metric.compute()

# Reset the metric for the next epoch 
metric.reset()
plot_errors(model, dataloader)
```
```python
# Using the same model, set the dropout probability to 0.8
model = nn.Sequential(
    nn.Linear(3072, 16),  # Adjust input size to match the reshaped tensor
    nn.ReLU(),            # ReLU activation function
    nn.Dropout(p=0.8)     # Dropout layer with 80% dropout probability
)
model(input_tensor)
```

```python
values = []
for idx in range(10):
    # Randomly sample a learning rate factor between 2 and 4
    factor = np.random.uniform(2, 4)
    lr = 10 ** -factor
    
    # Randomly select a momentum between 0.85 and 0.99
    momentum = np.random.uniform(0.85, 0.99)
    
    values.append((lr, momentum))
```
