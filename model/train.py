import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import io

# Check if MPS is available
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Read the CSV data into a pandas DataFrame
df = pd.read_csv("../dataset.csv", parse_dates=['date'],
                 date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d'))

# Add sinusoidally encoded date (time of year)
df['day_of_year'] = df['date'].dt.dayofyear
df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)

# Select features and target variable
features = df[['sunshine', 'global_radiation', 'max_temp', 'min_temp',
               'precipitation', 'pressure', 'snow_depth', 'sin_day_of_year', 'cos_day_of_year']]
target = df[['mean_temp', 'sunshine', 'precipitation']]

# Scale the features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

features_scaled = scaler_X.fit_transform(features)
target_scaled = scaler_y.fit_transform(target)

# Function to create sequences
def create_sequences(features, target, seq_length):
    xs = []
    ys = []
    for i in range(len(features) - seq_length):
        x = features[i:(i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 14  # Sequence length
X, y = create_sequences(features_scaled, target_scaled, seq_length)

# Create Dataset and DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        x = torch.tensor(self.sequences[idx], dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

dataset = TimeSeriesDataset(X, y)
batch_size = 5
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Define the LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=3, output_size=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Initialize hidden and cell states with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])

        return out

input_size = features.shape[1]  # Number of features
hidden_size = 32
num_layers = 2
output_size = target.shape[1]  # Updated to match the number of target variables

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

# Training the Model
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 25

for epoch in range(num_epochs):
    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), "model.npz")

# Making Predictions and Saving Outputs
model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        predictions.append(outputs.cpu().numpy())

# Concatenate predictions
predictions = np.concatenate(predictions, axis=0)

# Inverse transform the predictions and targets
predictions = scaler_y.inverse_transform(predictions)
targets = scaler_y.inverse_transform(y)

# Save predictions and targets to CSV
output_df = pd.DataFrame(
    predictions, columns=['Prediction_Mean_Temp', 'Prediction_Sunshine', 'Prediction_Precipitation']
)
output_df['Target_Mean_Temp'] = targets[:, 0]
output_df['Target_Sunshine'] = targets[:, 1]
output_df['Target_Precipitation'] = targets[:, 2]
output_df.to_csv('predictions.csv', index=False)
print("Predictions and targets saved to 'predictions.csv'.")

# Optionally, print the output
print(output_df)
