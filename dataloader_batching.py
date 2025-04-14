import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import time

def main():
    # Generating synthetic data
    X = torch.randn(1000, 100)
    y = torch.randint(0, 2, (1000, 1), dtype=torch.float32)

    # Dataset and DataLoader
    dataset = TensorDataset(X, y)
    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Hyperparameters
    input_dim = 100
    epochs = 5

    # Define model
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.model = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            return self.model(x)

    model = SimpleNN(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # Training loop with timing
    print("Starting training with DataLoader...")
    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for x_batch, y_batch in dataloader:
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * x_batch.size(0)

        avg_loss = epoch_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

    end_time = time.time()
    print(f"Training completed in {end_time - start_time:.2f} seconds.")

if __name__ == '__main__':
    main()