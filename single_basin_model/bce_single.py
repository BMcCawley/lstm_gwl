import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Assuming you have three numpy arrays: precipitation, PET, and groundwater_level
# Concatenate the features
features = np.column_stack((precip, pet))

# Normalize the features
scaler = MinMaxScaler(feature_range=(-1, 1))
features = scaler.fit_transform(features)

# Convert to PyTorch tensors
features = torch.from_numpy(features).float()
groundwater_level = torch.from_numpy(gwl).float()

# Split into training and test sets
(
    features_train,
    features_test,
    groundwater_level_train,
    groundwater_level_test,
) = train_test_split(features, groundwater_level, test_size=0.2, random_state=42)


# Define the LSTM model
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model, loss function, and optimizer
model = LSTM(input_size=2, hidden_size=50, num_layers=2, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Move data to the device
features_train = features_train.to(device)
groundwater_level_train = groundwater_level_train.to(device)
features_test = features_test.to(device)
groundwater_level_test = groundwater_level_test.to(device)

# Training loop
for epoch in range(100):  # 100 epochs
    model.train()
    optimizer.zero_grad()
    outputs = model(features_train)
    loss = criterion(outputs, groundwater_level_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

# Testing
model.eval()
with torch.no_grad():
    predictions = model(features_test)

# Print the first 5 predictions
print(predictions[:5])


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f"The model has {count_parameters(model)} parameters")
