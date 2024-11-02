import pandas as pd
import numpy as np
import torch
from torch import nn
from flask import Flask, jsonify, request
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta
from flask_cors import CORS

# Load the trained model and scalers
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, output_size=3):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# Instantiate the model and load weights
model = LSTMModel(input_size=9, hidden_size=32, num_layers=2, output_size=3).to(device)
model.load_state_dict(torch.load("model.something", map_location=device))
model.eval()

# Load dataset and scalers
df = pd.read_csv("../dataset.csv", parse_dates=['date'], date_parser=lambda x: pd.to_datetime(x, format='%Y%m%d'))
df['day_of_year'] = df['date'].dt.dayofyear
df['sin_day_of_year'] = np.sin(2 * np.pi * df['day_of_year'] / 365.0)
df['cos_day_of_year'] = np.cos(2 * np.pi * df['day_of_year'] / 365.0)

# Select features and target variable
features = df[['sunshine', 'global_radiation', 'max_temp', 'min_temp', 'precipitation', 'pressure', 'snow_depth', 'sin_day_of_year', 'cos_day_of_year']]
target = df[['mean_temp', 'sunshine', 'precipitation']]

# Scale the features and target
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
features_scaled = scaler_X.fit_transform(features)
target_scaled = scaler_y.fit_transform(target)


# Helper function to create sequences for model input
def create_sequences(features, target, seq_length, end_idx):
    xs = []
    ys = []
    for i in range(end_idx - seq_length, end_idx):
        x = features[i:(i + seq_length)]
        y = target[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


app = Flask(__name__)
CORS(app)

def get_week_dates(date_str):
    # Convert the input string to a date
    target_date = datetime.strptime(date_str, '%Y-%m-%d')

    # Find the Monday of the week containing the target date
    start_of_week = target_date - timedelta(days=target_date.weekday())

    # Generate the dates for the week, Monday to Sunday
    week_dates = [(start_of_week + timedelta(days=i)).date() for i in range(7)]

    return [date.strftime("%Y-%m-%d") for date in week_dates]


def get_day_of_week(date_str):
    # Convert the input string to a date
    date = datetime.strptime(date_str, '%Y-%m-%d')

    # Return the day of the week (e.g., 'Monday')
    return date.strftime('%A')


# Updated preprocess_and_predict with fuzzy date matching
def preprocess_and_predict(date_str, offset_delta=timedelta(days=0)):
    target_date = datetime.strptime(date_str, '%Y-%m-%d') + offset_delta

    # Find the closest date in the dataset
    if target_date not in df['date'].values:
        # Find the closest previous date
        closest_date = df[df['date'] <= target_date]['date'].max()
        if pd.isnull(closest_date):
            raise ValueError("No available data before the specified date")
        print(f"No exact match for {date_str}. Using closest date: {closest_date}")
    else:
        closest_date = target_date

    # Find the index of the closest date in the dataset
    end_idx = df.index[df['date'] == closest_date].tolist()[0]
    seq_length = 3  # Sequence length used in training

    if end_idx < seq_length:
        raise ValueError("Not enough data for the given sequence length")

    # Extract feature and target sequences up to the closest date
    features_seq, _ = create_sequences(features_scaled, target_scaled, seq_length, end_idx)

    # Reshape and make predictions
    inputs = torch.tensor(features_seq, dtype=torch.float32).to(device)
    with torch.no_grad():
        prediction = model(inputs).cpu().numpy()

    # Inverse transform to original scale
    prediction = scaler_y.inverse_transform(prediction[-1:])
    return prediction[0].tolist()


def make_response(prediction):
    return {
        'Prediction_Mean_Temp': prediction[0],
        'Prediction_Sunshine': prediction[1],
        'Prediction_Precipitation': prediction[2]
    }

@app.route('/predict', methods=['GET'])
def predict():
    date = request.args.get('date')
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400

    try:
        return jsonify(make_response(preprocess_and_predict(date)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict-week', methods=['GET'])
def predict_wee():
    date = request.args.get('starting')
    if not date:
        return jsonify({"error": "Date parameter is required"}), 400
    try:
        return jsonify({
            get_day_of_week(day): make_response(preprocess_and_predict(day))
            for day in get_week_dates(date)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run()
