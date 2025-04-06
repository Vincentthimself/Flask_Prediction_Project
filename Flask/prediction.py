# Import Libraries
import numpy as np 
import pandas as pd
import yfinance as yf
import seaborn as sns
import tensorflow as tf 
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
from flask import Blueprint, url_for, render_template, request
from flask_login import login_required, current_user
plt.style.use("fivethirtyeight")

prediction = Blueprint('prediction', __name__)

# Load the model (make sure your model is in the correct path)
model = load_model('Flask/lstm_model.h5')

@prediction.route('/prediction_page', methods=['GET', 'POST'])
def lstm_prediction():
    if request.method == 'POST':
        currency_pairs = request.form.get('currency_pairs')
        days_to_predict = request.form.get('days_to_predict')
        # prediction_days = int(request.form.get'prediction_days')
        if not currency_pairs:
            currency_pairs = 'EURUSD=X'  # Default stock if none is entered
        if not days_to_predict or not days_to_predict.isdigit() or int(days_to_predict) <= 0:
            return render_template('prediction.html', error="Please enter a valid number of days to predict.", user=current_user)
        days_to_predict = int(days_to_predict)
        
        # Download historical data
        data = yf.download(currency_pairs, start='2020-01-01', end=datetime.datetime.today().strftime('%Y-%m-%d'))
        # Focus on 'Close' prices
        data = data[['Close']].dropna()  # Drop NaN values if any
        if data.empty:
            return render_template('prediction.html', error="Invalid currency pair or no data available.", user=current_user)

        # data_desc = data.describe()   

        # Exponential Moving Averages
        ema20 = data.Close.ewm(span=20, adjust=False).mean()
        ema50 = data.Close.ewm(span=50, adjust=False).mean()

        try: 
            # Step 2 Scale the training data 
            scaler = joblib.load('Flask/minmax_scaler.pkl')
            new_scaled_data = scaler.transform(data)
       

            # Create empty lists for features 
            x_train = [] # features 
            y_train = [] # closing price

            # Creating sliding window
            # Populate x_train with 90 days of data and y_train with the following day's closing price   
            for i in range(90, len(new_scaled_data)):
                x_train.append(new_scaled_data[i-90:i, 0])
                y_train.append(new_scaled_data[i, 0])
            
            # Convert x_train and y_train to numpy arrays for models training
            x_train, y_train = np.array(x_train), np.array(y_train)

            # Reshape x_train to format [samples, time steps, features] required for LSTM model
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

            # Predict future prices for the specified number of days
            future_predictions = []
            last_90_days = new_scaled_data[-90:]

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

            # Plotting future predictions with the last month of historical data
            plt.figure(figsize=(12, 6))

            # Determine the start date for the last month of data
            last_month_start = data.index[-1] - pd.DateOffset(days=30)


            fig1, ax1 = plt.subplots(figsize=(12,8))
            ax1.plot(data.Close, 'y', label='Closing Price')
            ax1.plot(ema20, 'g', label='EMA 20')
            ax1.plot(ema50, 'r', label='EMA 50')
            ax1.set_title("Closing Price vs TIME (20 and 50 Days EMA)")
            ax1.set_xlabel("Time")
            ax1.set_ylabel("Price")
            ax1.legend()
            ema_chart_path = "Flask/static/ema_20_50.png"
            fig1.savefig(ema_chart_path)
            plt.close(fig1)



            fig2, ax2 = plt.subplots(figsize=(12,8))
            ax2.plot(data.loc[last_month_start:, 'Close'], label='Last Month Historical Price')
            ax2.plot(future_df['Predicted Price'], label='Future Predicted Price', linestyle='--')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m-%y'))
            ax2.grid(True)
            ax2.set_title('Stock Price Prediction with Future Forecast')
            ax2.set_xlabel('Date')
            ax2.set_ylabel('Price')
            ax2.legend()
            currency_prediction = "Flask/static/currency_prediction.png"
            fig2.savefig(currency_prediction)
            plt.close(fig2)
            # Enable interactive tooltips
            mplcursors.cursor(hover=True)


        except FileNotFoundError:
            return render_template('prediction.html', error="Scaler file not found. Please train the model first.", user=current_user)
        except Exception as e:
            return render_template('prediction.html', error="An error occurred during prediction: " + str(e), user=current_user)
        return render_template('prediction.html', 
                               currency_pairs=currency_pairs, 
                               plot_path_ema_20_50=ema_chart_path,
                               plot_currency_prediction=currency_prediction,
                               days_to_predict=days_to_predict, 
                               future_predictions=future_predictions, 
                               data=data,
                               user=current_user)       
        
    return render_template('prediction.html', user=current_user)
       
