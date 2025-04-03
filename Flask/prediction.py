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
from flask import Blueprint, url_for, render_template, request
from flask_login import login_required, current_user

prediction = Blueprint('prediction', __name__)

# Load the model (make sure your model is in the correct path)
model = load_model('Flask/lstm_model.h5')

@prediction.route('/prediction_page', methods=['GET', 'POST'])
def lstm_prediction():
    if request.method == 'POST':
        currency_pairs = request.form.get('currency_pairs')
        # prediction_days = int(request.form.get'prediction_days')
        if not currency_pairs:
            currency_pairs = 'EURUSD=X'  # Default stock if none is entered

    # Download historical data

        end_date = '2025-02-28'
        start_date = '2020-02-29'

        data = yf.download(currency_pairs, start=start_date, end=end_date)
        # Focus on 'Close' prices
        data = data[['Close']].dropna()  # Drop NaN values if any
        if data.empty:
            return render_template('prediction_page.html', error="Invalid currency pair or no data available.")

        data_desc = data.describe()   

        # Exponential Moving Averages
        ema20 = data.Close.ewm(span=20, adjust=False).mean()
        ema50 = data.Close.ewm(span=50, adjust=False).mean()


        # Step 2 Prepare training and testing datasets
        training_data_len = int(np.ceil(len(data) * 0.7))

        # Split the data into training and test sets
        raw_train_data = data[:training_data_len] # 70% 
        raw_test_data = data[training_data_len:] # 30% 




    return render_template('home.html', 
                           user=current_user)
        
        # scaler = MinMaxScaler(feature_range=(0, 1))
        # scaled_data = scaler.fit_transform(data)
       
