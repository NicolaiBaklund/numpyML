# NumPy Machine Learning Framework
To teach myself different concepts in machine learning, I am building a NumPy based dynamic ML framework from scratch. The framework will grow gradually as I learn and implement concepts. I am copying ideas from popular ML frameworks like PyTorch, but the implementation will be my own. My current end-goal is to implement a Transformer using this framework.

## Sequential Model
The framework currently supports a simple sequential model.

### Layers
- Dense (Fully Connected) Layer
- Dropout Layer
- Flatten Layer
- Activation Layers: ReLU and Sigmoid

### Loss Functions
- Mean Squared Error (MSE)
- Cross-Entropy Loss on logits
- BCE on logits
- Huber (For RL-DQN)

### Optimizers
- plain SGD
- Adam

### Example
```python
# Comparison of my framework with PyTorch
# Using NumPyML
import numpy as np
from numpyml.nn import model, layers, losses, optimizers, activations
model = model.Sequential([
    layers.Dense(2, 25),
    activations.ReLU(),
    layers.Dense(25, 10),
    activations.ReLU(),
    layers.Dense(10, 1),
    activations.Sigmoid()
])

loss_fn = losses.BinaryCrossEntropyWithLogits()
# optimizer needs both model parameters and gradients
optim = optimizers.SGD(model.parameters(), model.gradients(), learning_rate=0.01)

# Training loop numpyML:
for epoch in range(100):
    model.train()
    # Forward pass
    outputs = model.forward(X_train)
    loss = loss_fn.forward(outputs, y_train)

    # Backward pass
    optim.zero_grad()
    loss_grad = loss_fn.backward()
    model.backward(loss_grad)

    # Update parameters
    optim.step()

    model.eval()
    # Calculate validation loss



# Using PyTorch
import torch.nn as nn
import torch.optim as optim

model = nn.Sequential(
    nn.Linear(2, 25),
    nn.ReLU(),
    nn.Linear(25, 10),
    nn.ReLU(),
    nn.Linear(10, 1),
    nn.Sigmoid()
)
loss_fn = nn.BCEWithLogitsLoss()
optim = optim.SGD(model.parameters(), lr=0.01)


# Training loop PyTorch:
for epoch in range(100):
    model.train()
    # Forward pass
    outputs = model(X_train)
    loss = loss_fn(outputs, y_train)

    # Backward pass
    optim.zero_grad()
    loss.backward()

    # Update parameters
    optim.step()

    model.eval()
    # Calculate validation loss
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = loss_fn(val_outputs, y_val)

Both implementations follow a similar structure, demonstrating the usability of the NumPyML framework.
```