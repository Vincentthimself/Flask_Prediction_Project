import numpy as np 
import pandas as pd
import yfinance as yf
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
model = load_model('lstm_model.h5')

@prediction.route('/prediction_page', methods=['GET', 'POST'])
def lstm_prediction():
    if request.method == 'POST':
        currency_pairs = request.form.get('currency_pairs')
        # prediction_days = int(request.form.get'prediction_days')
        if not currency_pairs:
            currency_pairs = 'EURUSD=X'  # Default stock if none is entered

    # Download historical data
        data = yf.download(currency_pairs, start='2020-01-01', end=datetime.datetime.today().strftime('%Y-%m-%d'))
        # Focus on 'Close' prices
        data = data[['Close']].dropna()  # Drop NaN values if any
        if data.empty:
            return render_template('prediction_page.html', error="Invalid currency pair or no data available.")
        

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
       
