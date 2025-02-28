import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

# Define edges (source node to target node)
edge_index = torch.tensor([[0, 1, 1, 2, 2, 3],  # Source nodes
                           [1, 0, 2, 1, 3, 2]], # Target nodes
                          dtype=torch.long)

# Define node features (each node has a 2D feature vector)
x = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=torch.float)

# Define labels (for classification)
y = torch.tensor([0, 1, 0, 1], dtype=torch.long)

# Wrap it in a PyG Data object
data = Data(x=x, edge_index=edge_index, y=y)

class GNN(torch.nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.conv1 = GCNConv(2, 4)  # Input features = 2, Hidden layer = 4
        self.conv2 = GCNConv(4, 2)  # Hidden layer = 4, Output classes = 2

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)  # Log-softmax for classification

# Initialize model, optimizer, and loss function
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()  # Negative log-likelihood loss for classification

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)  # Forward pass
    loss = loss_fn(out, data.y)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Initialize model, optimizer, and loss function
model = GNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = torch.nn.NLLLoss()  # Negative log-likelihood loss for classification

# Training loop
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)  # Forward pass
    loss = loss_fn(out, data.y)  # Compute loss
    loss.backward()  # Backpropagation
    optimizer.step()  # Update weights
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
