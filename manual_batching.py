import torch
import torch.nn as nn
import torch.optim as optim
import time

# Generating synthetic data
X = torch.randn(1000, 100)
y = torch.randint(0, 2, (1000, 1), dtype=torch.float32)

# Defining hyperparameters
input_dim = 100
batch_size = 64
epochs = 5

# Defining the model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, X):
        return self.model(X)
        
# Instantiating model
model = SimpleNN(input_dim)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

start = time.time()
print("Starting training...")

# Training loop (manual batching)
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0

    for i in range(0, len(X), batch_size):
        x_batch = X[i:i+batch_size]
        y_batch = y[i:i+batch_size]

        outputs = model(x_batch)
        loss = criterion(outputs, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(X):.4f}")

end = time.time()
print(f"Training completed in {end - start:.2f} seconds.")