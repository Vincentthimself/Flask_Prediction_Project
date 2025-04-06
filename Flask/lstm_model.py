# Import Libraries
import numpy as np 
import pandas as pd
import yfinance as yf
import tensorflow as tf 
import seaborn as sns
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import datetime
import os
import matplotlib.dates as mdates
import mplcursors
import joblib

# Import to ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Step 1 Download data
ticker = 'EURUSD=X'

end_date = '2025-02-28'
start_date = '2020-02-29'  # Exactly 5 years before the end date

data = yf.download(ticker, start=start_date, end=end_date)
# data = yf.download(ticker, start='2020-01-01', end=datetime.datetime.today().strftime('%Y-%m-%d'))

# Handle missing data
data = data.dropna()

# Focus on 'Close' prices for simplicity 
data = data[['Close']]
# print(data.describe())
# print(data.info())
# print(data.isnull().sum())


# # Plot the 'Close' prices
# plt.figure(figsize=(18, 9))
# plt.plot(data.index, data['Close'], label='Close Price')
# plt.xlabel('Date', fontsize=18)
# plt.ylabel('Close Price', fontsize=18)
# plt.title('Close Price Over Time', fontsize=20)
# plt.legend()
# plt.show()

# Plot correlation between features
# plt.figure(figsize=(12, 9))
# sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
# plt.title("Feature Correlation Heatmap")
# plt.show()


# Step 2 Prepare training and testing datasets
training_data_len = int(np.ceil(len(data) * 0.7))

# Split the data into training and test sets
raw_train_data = data[:training_data_len] # 70% 
raw_test_data = data[training_data_len:] # 30% 
# print(train_data)
# print(test_data)

# Step 3 Scale the training data 
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(raw_train_data)
print(scaled_train_data.shape)

# Scale the test data using the same scaler fitted on the training data
scaled_test_data = scaler.transform(raw_test_data)
# print(scaled_test_data)
# Save the scaler
joblib.dump(scaler, 'minmax_scaler.pkl')

# Create empty lists for features 
x_train = [] # features 
y_train = [] # closing price

# Creating sliding window
# Populate x_train with 90 days of data and y_train with the following day's closing price   
for i in range(90, len(scaled_train_data)):
    x_train.append(scaled_train_data[i-90:i, 0])
    y_train.append(scaled_train_data[i, 0])

# Convert x_train and y_train to numpy arrays for models training
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape x_train to format [samples, time steps, features] required for LSTM model
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))



# # Check if the model file exists
# model_file = 'lstm_model.h5'
# if not os.path.exists(model_file):
#     # Step 4 Build the LSTM model
#     model = Sequential([
#         LSTM(100, return_sequences=True, input_shape=(x_train.shape[1], 1)),
#         Dropout(0.1),
#         LSTM(100, return_sequences=False),
#         Dropout(0.1),
#         Dense(25),
#         Dense(1)
#     ])


model_file = 'lstm_model.h5'
if not os.path.exists(model_file):
    model = Sequential()

    # First layer
    model.add(LSTM(256, activation="relu", return_sequences=True, input_shape=(x_train.shape[1], 1)))

    # Second layer
    model.add(LSTM(256, activation="relu", return_sequences=False))

    # Third layer (Dense)
    model.add(Dense(128, activation="relu"))

    # Fourth layer (Dropout)
    model.add(Dropout(0.2))

    # Fifth Output layer
    model.add(Dense(1))

    model.summary()

    model.compile(optimizer='adam', loss='mean_squared_error')

    history = model.fit(
            x_train, 
            y_train, 
            batch_size=64, 
            epochs=30,
    )

    model.save(model_file)
else: 
    model = load_model(model_file)


# # # Compile the model
# # model.compile(optimizer='adam', loss='mean_squared_error')

# # # Step 5 Train the model without early stopping
# # history = model.fit(
# #         x_train, 
# #         y_train, 
# #         batch_size=150, 
# #         epochs=20,
# #         validation_split=0.1,
# #         verbose=1
# # )

# #     # Save the model
# #     model.save(model_file)

# #     # Plot training history
# #     plt.figure(figsize=(12,6))
# #     plt.plot(history.history['loss'], label='Training Loss')
# #     plt.plot(history.history['val_loss'], label='Validation Loss')
# #     plt.title('Model Loss Progress')
# #     plt.xlabel('Epochs')
# #     plt.ylabel('Loss')
# #     plt.legend()
# #     plt.show()
# # else:
# #     # Load the model
# #     model = load_model(model_file)

# Create the test dataset
test_data = np.vstack([scaled_train_data[-90:], scaled_test_data])
print(f"Length of test_data: {len(test_data)}")


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

# # Prompt user for number of days to predict
# while True:
#     try:
#         days_to_predict = int(input("Enter the number of days to predict (1-5): "))
#         if 1 <= days_to_predict <= 5:
#             break
#         else:
#             print("Please enter a number between 1 and 5.")
#     except ValueError:
#         print("Invalid input. Please enter a number between 1 and 5.")

# # Predict future prices for the specified number of days
# future_predictions = []
# last_90_days = test_data[-90:]

# for _ in range(days_to_predict):
#     last_90_days_reshaped = last_90_days.reshape((1, last_90_days.shape[0], 1))
#     next_day_prediction = model.predict(last_90_days_reshaped)
#     future_predictions.append(next_day_prediction[0, 0])
#     last_90_days = np.append(last_90_days[1:], next_day_prediction)

# # future_predictions = np.array(future_predictions).reshape(-1, 1)
# # future_predictions = scaler.inverse_transform(future_predictions)

# # # Create a new DataFrame for future predictions
# # future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=days_to_predict)
# # future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted Price'])

# # # Plotting future predictions with the last month of historical data
# # plt.figure(figsize=(12, 6))

# # # Determine the start date for the last month of data
# # last_month_start = data.index[-1] - pd.DateOffset(days=30)

# # # Plot the last month of historical data
# # plt.plot(data.loc[last_month_start:, 'Close'], label='Last Month Historical Price')

# # # Plot the future predictions
# # plt.plot(future_df['Predicted Price'], label='Future Predicted Price', linestyle='--')

# # # Format the x-axis to show concise dates
# # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))

# # # Add grid
# # plt.grid(True)

# # plt.title('Stock Price Prediction with Future Forecast')
# # plt.xlabel('Date')
# # plt.ylabel('Price')
# # plt.legend()

# # # Enable interactive tooltips
# # mplcursors.cursor(hover=True)

# # plt.show()

# Print metrics after the entire process
print('\nModel Performance Metrics:')
print('Mean Absolute Error:', mean_absolute_error(valid['Close'], valid['Predictions']))
print('Root Mean Squared Error:', np.sqrt(mean_squared_error(valid['Close'], valid['Predictions'])))
print('R-squared:', r2_score(valid['Close'], valid['Predictions']))
print(valid)



