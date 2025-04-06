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
from flask import Blueprint, url_for, render_template, request, redirect
from flask_login import login_required, current_user
import skfuzzy as fuzz
from skfuzzy import control as ctrl
plt.style.use("fivethirtyeight")

prediction = Blueprint('prediction', __name__)

# Load the model (make sure your model is in the correct path)
model = load_model('Flask/lstm_model.h5')

@prediction.route('/prediction_page', methods=['GET', 'POST'])
def lstm_prediction():
    if request.method == 'POST':
        currency_pairs = request.form.get('currency_pairs')
        days_to_predict = request.form.get('days_to_predict')
        if not currency_pairs:
            currency_pairs = 'EURUSD=X'  # Default stock if none is entered
        if not days_to_predict or not days_to_predict.isdigit() or int(days_to_predict) <= 0:
            return render_template('prediction.html', error="Please enter a valid number of days to predict.", user=current_user)
        days_to_predict = int(days_to_predict)
        
        # Download historical data
        data = yf.download(currency_pairs, start='2020-01-01', end=datetime.datetime.today().strftime('%Y-%m-%d'))
        data = data[['Close']].dropna()  # Drop NaN values if any
        if data.empty:
            return render_template('prediction.html', error="Invalid currency pair or no data available.", user=current_user)

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

            # Convert DataFrame to a list of dictionaries for the template
            future_predictions_list = [
                {'date': date.strftime('%Y-%m-%d'), 'Predicted Price': price[0]}
                for date, price in zip(future_df.index, future_df.values)
            ]

            # Plotting future predictions with the last month of historical data
            plt.figure(figsize=(12, 6))

            # Determine the start date for the last month of data
            last_month_start = data.index[-1] - pd.DateOffset(days=30)

            fig2, ax2 = plt.subplots(figsize=(12,8))
            ax2.plot(data.loc[last_month_start:, 'Close'], label='Last Month Historical Price')
            ax2.plot(future_df['Predicted Price'], label='Future Predicted Price', linestyle='--')
            ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d-%m'))
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




            # Fuzzy Logic Integration
            # Compute MACD
            exp12 = data['Close'].ewm(span=12, adjust=False).mean()
            exp26 = data['Close'].ewm(span=26, adjust=False).mean()
            macd_line = exp12 - exp26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()
            data['MACD_hist'] = macd_line - signal_line

            # Compute RSI
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(14).mean()
            avg_loss = loss.rolling(14).mean()
            rs = avg_gain / avg_loss
            data['RSI'] = 100 - (100 / (1 + rs))

            # Compute Stochastic Oscillator
            low14 = data['Close'].rolling(14).min()
            high14 = data['Close'].rolling(14).max()
            data['%K'] = (data['Close'] - low14) / (high14 - low14) * 100

            # Define fuzzy variables
            macd_range = np.arange(-1, 1, 0.001)
            rsi_range = np.arange(0, 100, 0.1)
            stochastic_range = np.arange(0, 100, 0.1)
            action_range = np.arange(-1, 1.01, 0.01)

            macd = ctrl.Antecedent(macd_range, 'macd')
            rsi = ctrl.Antecedent(rsi_range, 'rsi')
            stochastic = ctrl.Antecedent(stochastic_range, 'stochastic')
            action = ctrl.Consequent(action_range, 'action')

            # Define membership functions
            macd['low'] = fuzz.trimf(macd_range, [-1, -0.5, 0])
            macd['high'] = fuzz.trimf(macd_range, [0, 0.5, 1])

            rsi['low'] = fuzz.trimf(rsi_range, [0, 15, 35])
            rsi['medium'] = fuzz.trimf(rsi_range, [30, 50, 70])
            rsi['high'] = fuzz.trimf(rsi_range, [65, 85, 100])

            stochastic['low'] = fuzz.trimf(stochastic_range, [0, 10, 30])
            stochastic['medium'] = fuzz.trimf(stochastic_range, [25, 50, 75])
            stochastic['high'] = fuzz.trimf(stochastic_range, [70, 90, 100])

            action['sell'] = fuzz.trimf(action_range, [-1, -1, -0.5])
            action['hold'] = fuzz.trimf(action_range, [-0.5, 0, 0.5])
            action['buy'] = fuzz.trimf(action_range, [0.5, 1, 1])

            # Define fuzzy rules
            rules = [
                ctrl.Rule(macd['low'] & rsi['low'] & stochastic['low'], action['buy']),
                ctrl.Rule(macd['low'] & rsi['low'] & stochastic['medium'], action['buy']),
                ctrl.Rule(macd['low'] & rsi['low'] & stochastic['high'], action['hold']),
                ctrl.Rule(macd['low'] & rsi['medium'] & stochastic['low'], action['hold']),
                ctrl.Rule(macd['low'] & rsi['medium'] & stochastic['medium'], action['hold']),
                ctrl.Rule(macd['low'] & rsi['medium'] & stochastic['high'], action['sell']),
                ctrl.Rule(macd['low'] & rsi['high'] & stochastic['low'], action['sell']),
                ctrl.Rule(macd['low'] & rsi['high'] & stochastic['medium'], action['sell']),
                ctrl.Rule(macd['low'] & rsi['high'] & stochastic['high'], action['sell']),
                ctrl.Rule(macd['high'] & rsi['low'] & stochastic['low'], action['buy']),
                ctrl.Rule(macd['high'] & rsi['low'] & stochastic['medium'], action['buy']),
                ctrl.Rule(macd['high'] & rsi['low'] & stochastic['high'], action['buy']),
                ctrl.Rule(macd['high'] & rsi['medium'] & stochastic['low'], action['hold']),
                ctrl.Rule(macd['high'] & rsi['medium'] & stochastic['medium'], action['hold']),
                ctrl.Rule(macd['high'] & rsi['medium'] & stochastic['high'], action['sell']),
                ctrl.Rule(macd['high'] & rsi['high'] & stochastic['low'], action['sell']),
                ctrl.Rule(macd['high'] & rsi['high'] & stochastic['medium'], action['sell']),
                ctrl.Rule(macd['high'] & rsi['high'] & stochastic['high'], action['sell'])
            ]

            # Create fuzzy control system
            trading_ctrl = ctrl.ControlSystem(rules)
            trading_sim = ctrl.ControlSystemSimulation(trading_ctrl)

            # Generate signals
            signals = []
            for idx in range(len(data)):
                try:
                    trading_sim.input['macd'] = data['MACD_hist'].iloc[idx]
                    trading_sim.input['rsi'] = data['RSI'].iloc[idx]
                    trading_sim.input['stochastic'] = data['%K'].iloc[idx]
                    trading_sim.compute()
                    action_value = trading_sim.output['action']
                    if action_value <= -0.25:
                        signals.append('sell')
                    elif -0.25 < action_value <= 0.25:
                        signals.append('hold')
                    else:
                        signals.append('buy')
                except:
                    signals.append('hold')

            data['Signal'] = signals

        except FileNotFoundError:
            return render_template('prediction.html', error="Scaler file not found. Please train the model first.", user=current_user)
        except Exception as e:
            return render_template('prediction.html', error="An error occurred during prediction: " + str(e), user=current_user)
        return render_template('prediction.html', 
                               currency_pairs=currency_pairs, 
                               plot_currency_prediction=currency_prediction,
                               days_to_predict=days_to_predict, 
                               predictions=future_predictions_list,  # Pass the formatted predictions
                               signals=data[['Signal']].to_dict(orient='records'),  # Pass signals to the template
                               user=current_user)       
        
    return render_template('prediction.html', user=current_user)

