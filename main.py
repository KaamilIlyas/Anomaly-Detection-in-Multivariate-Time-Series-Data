import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Load the dataset
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
test_labels = pd.read_csv("test_label.csv")

# Preprocess the data
train_data.dropna(inplace=True)
test_data.dropna(inplace=True)
test_labels.dropna(inplace=True)

# Convert timestamp column to datetime
train_data['timestamp_(min)'] = pd.to_datetime(train_data['timestamp_(min)'], unit='m')
test_data['timestamp_(min)'] = pd.to_datetime(test_data['timestamp_(min)'], unit='m')
test_labels['timestamp_(min)'] = pd.to_datetime(test_labels['timestamp_(min)'], unit='m')

# Set timestamp column as index
train_data.set_index('timestamp_(min)', inplace=True)
test_data.set_index('timestamp_(min)', inplace=True)
test_labels.set_index('timestamp_(min)', inplace=True)

# Data Augmentation with geometric distribution masks
def geometric_mask_augmentation(data):

   # Apply data augmentation with geometric distribution masks.
    mask = np.random.geometric(p=0.1, size=data.shape)
    augmented_data = data + mask * np.random.normal(0, 0.1, size=data.shape)
    return augmented_data

train_data_augmented = geometric_mask_augmentation(train_data.values)

# Convert data to PyTorch tensors
train_data_tensor = torch.tensor(train_data_augmented, dtype=torch.float32)
train_labels_tensor = torch.tensor(train_data.values, dtype=torch.float32)
test_data_tensor = torch.tensor(test_data.values, dtype=torch.float32)

# Define the Transformer-based Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size):
        super(Autoencoder, self).__init__()

        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_size)
        )

    def forward(self, x):
        # Forward pass through encoder
        encoded = self.encoder(x)

        # Forward pass through decoder
        decoded = self.decoder(encoded)
        return decoded

# Initialize the Autoencoder model
input_size = train_data_tensor.shape[1]
autoencoder = Autoencoder(input_size)

# Define the Contrastive Loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x1, x2):
        # Flatten input tensors
        x1 = torch.flatten(x1, start_dim=1)
        x2 = torch.flatten(x2, start_dim=1)

        # Compute cosine similarity
        sim = torch.cosine_similarity(x1, x2, dim=-1) / self.temperature

        # Compute contrastive loss
        loss = -torch.log(torch.exp(sim).sum(dim=-1) / (torch.exp(sim).sum(dim=-1) + torch.exp(-sim).sum(dim=-1)))
        return loss.mean()

# Initialize the contrastive model
contrastive_model = Autoencoder(input_size)

# Train the contrastive model
criterion = ContrastiveLoss()
optimizer = optim.Adam(contrastive_model.parameters(), lr=0.001)

train_dataset = TensorDataset(train_data_tensor, train_data_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

num_epochs = 10
for epoch in range(num_epochs):
    contrastive_model.train()
    total_loss = 0
    for data in train_loader:
        inputs, _ = data
        optimizer.zero_grad()
        outputs = contrastive_model(inputs)
        loss = criterion(inputs, outputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

# Evaluate the model on the test dataset
test_predictions = contrastive_model(test_data_tensor)
mse = mean_squared_error(test_data_tensor.detach().numpy(), test_predictions.detach().numpy())
print("Mean Squared Error on Test Data:", mse)

# Detect anomalies
threshold = np.mean(mse) + 2 * np.std(mse)
anomalies = mse > threshold
print("Number of Anomalies Detected:", np.sum(anomalies))