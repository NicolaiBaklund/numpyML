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

