# 1. Prepare data as PyTorch tensors (ideally with dtype torch.float32 and requires_grad only for learnable parameters).
# 2. Define a model (subclass nn.Module or use raw torch.Tensor parameters).
# 3. Forward pass: use your model to get predictions.
# 4. Compute loss using a built-in criterion or a custom function.
# 5. Zero gradients on your parameters/optimizers.
# 6. Backward pass (loss.backward()) to populate .grad.
# 7. Update parameters (manually with param.data -= lr * param.grad and with torch.no_grad(), or via an optimizer’s .step()).
# 8. Repeat for each epoch/batch.


import torch
import torch.nn as nn
import torch.optim as optim


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()

        # Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Activation
        self.relu = nn.ReLU()

        # Layer 2
        self.fc2 = nn.Linear(hidden_size, output_size)

    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x



# Instantiate model
input_size = 10
output_size = 1
model = MLP(input_size, 50, output_size)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=1e-3)

# Toy dataset of 100 samples
samples = torch.randn(100, 10) #100 samples of dim=10
targets = torch.randn(100, 1) #corresponding targets 

# Parameters for training 
num_epochs = 20
batch_size = 16


# Training
for epoch in range(num_epochs):
    permutation = torch.randperm(samples.size(0))
    epoch_loss = 0.0

    for i in range(0, samples.size(0), batch_size):
        # Get batch indices
        indices = permutation[i:i+batch_size]
        batch_samples = samples[indices]
        batch_targets = targets[indices]

        # Forward pass
        preds = model(batch_samples)
        loss = criterion(preds, batch_targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / (samples.size(0) / batch_size)
    print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")