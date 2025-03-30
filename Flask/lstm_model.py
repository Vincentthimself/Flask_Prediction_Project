# Import Libraries
import numpy as np 
import pandas as pd
import yfinance as yf
import tensorflow as tf 
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import datetime
import os
import matplotlib.dates as mdates
import mplcursors

# Import to ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Step 1 Download data
ticker = 'EURUSD=X'
data = yf.download(ticker, start='2020-01-01', end=datetime.datetime.today().strftime('%Y-%m-%d'))

# Focus on 'Close' prices for simplicity 
data = data[['Close']]

# Step 2 Preprocess data 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Step 3 Prepare training data
training_data_len = int(np.ceil(len(scaled_data) * 0.7))

# Split the scaled data into training set
train_data = scaled_data[0:training_data_len, :]

# Create empty lists for features 
x_train = []
y_train = []

# Populate x_train with 90 days of data and y_train with the following day's closing price   
for i in range(90, len(train_data)):
    x_train.append(train_data[i-90:i, 0])
    y_train.append(train_data[i, 0])

# Convert x_train and y_train to numpy arrays for models training
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape x_train to format [samples, time steps, features] required for LSTM model
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

# Check if the model file exists
model_file = 'lstm_model.h5'
if not os.path.exists(model_file):
    # Step 4 Build the LSTM model
    model = Sequential([
        LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)),
        Dropout(0.1),
        LSTM(100, return_sequences=False),
        Dropout(0.1),
        Dense(25, activation='relu'),
        Dense(1)
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Step 5 Train the model without early stopping
    history = model.fit(
        x_train, 
        y_train, 
        batch_size=150, 
        epochs=20,
        validation_split=0.1,
        verbose=1
    )

    # Save the model
    model.save(model_file)

    # Plot training history
    plt.figure(figsize=(12,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
else:
    # Load the model
    model = load_model(model_file)

# Create the test dataset
test_data = scaled_data[training_data_len - 90:, :]

# Create x_test and y_test
x_test = []
y_test = data['Close'][training_data_len:].values

for i in range(90, len(test_data)):
    x_test.append(test_data[i-90:i, 0])

# Convert to numpy array
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Get predicted prices
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Create DataFrames for visualization
train = data[:training_data_len]
valid = data[training_data_len:]
valid.loc[:, 'Predictions'] = predicted_prices

# Plotting
plt.figure(figsize=(12,6))
plt.plot(train['Close'], label='Training Data')
plt.plot(valid['Close'], label='Real Price')
plt.plot(valid['Predictions'], label='Predicted Price', linestyle='--')
plt.title('Stock Price Prediction')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Prompt user for number of days to predict
while True:
    try:
        days_to_predict = int(input("Enter the number of days to predict (1-5): "))
        if 1 <= days_to_predict <= 5:
            break
        else:
            print("Please enter a number between 1 and 5.")
    except ValueError:
        print("Invalid input. Please enter a number between 1 and 5.")

# Predict future prices for the specified number of days
future_predictions = []
last_90_days = test_data[-90:]

for _ in range(days_to_predict):
    last_90_days_reshaped = last_90_days.reshape((1, last_90_days.shape[0], 1))
    next_day_prediction = model.predict(last_90_days_reshaped)
    future_predictions.append(next_day_prediction[0, 0])
    last_90_days = np.append(last_90_days[1:], next_day_prediction)

future_predictions = np.array(future_predictions).reshape(-1, 1)
future_predictions = scaler.inverse_transform(future_predictions)

# Create a new DataFrame for future predictions
future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)
future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Price'])

# Plotting future predictions with the last 7 days of historical data
plt.figure(figsize=(12, 6))

# Determine the start date for the last 7 days of data
last_week_start = data.index[-1] - pd.DateOffset(days=6)

# Plot the last 7 days of historical data
plt.plot(data.loc[last_week_start:, 'Close'], label='Last Week Historical Price')

# Plot the future predictions with markers
plt.plot(future_df['Predicted Price'], label='Future Predicted Price', linestyle='--', marker='o')

# Format the x-axis to show concise dates
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

# Add grid
plt.grid(True)

plt.title('Stock Price Prediction with Future Forecast')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Enable interactive tooltips
mplcursors.cursor(hover=True)

plt.show()

# Print metrics after the entire process
print('\nModel Performance Metrics:')
print('Mean Absolute Error:', mean_absolute_error(valid['Close'], valid['Predictions']))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(valid['Close'], valid['Predictions'])))
print('R-squared:', r2_score(valid['Close'], valid['Predictions']))


# batch normalization can work 



