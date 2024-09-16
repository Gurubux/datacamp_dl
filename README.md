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
model = nn.Sequential(  nn.Linear(10, 18),    
nn.Linear(18, 20),   
nn.Linear(20, 5))

"Activation function"  
Activation function as the last Layer      
Binary-Sigmoid classification    
Multiclass-Sofmax classification  
