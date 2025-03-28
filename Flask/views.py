from flask import Blueprint, url_for, render_template, request
from flask_login import login_required, current_user
import json, random 
import pandas as pd

import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')   
import matplotlib.dates as mdates
import os
from datetime import datetime, timedelta

views = Blueprint('views', __name__)

@views.route('/home_page')
@login_required
def home():
    return render_template("home.html", user=current_user)

# Define currency pairs
currency_pairs = {
    "EUR/USD": "EURUSD=X",
    "JPY/USD": "JPYUSD=X",
    "GBP/USD": "GBPUSD=X",
    "AUD/USD": "AUDUSD=X"
}

# Call API from Yahoo Finance
def get_forex_data(pair, start_date=None, end_date=None, interval=None, period=None):
    if interval:
        # Use the provided period; for intraday data, we use "5d" as it's valid.
        data = yf.download(pair, period=period if period else "5d", interval=interval)
    else:
        data = yf.download(pair, start=start_date, end=end_date)
    if not data.empty:
        return data
    else:
        return None

@views.route('/chart_page', methods=["GET", "POST"])
def index():
    plot_path = None
    selected_pair = None
    selected_timeframe = None

    if request.method == "POST":
        # Get the selected currency pair and timeframe from the form.
        selected_pair = request.form.get("currency_pair")
        selected_timeframe = request.form.get("timeframe")
        
        forex_data = None

    if selected_pair and selected_timeframe:
            ticker = currency_pairs[selected_pair]

            # Check if the timeframe is intraday (1 minute, 5 minutes, 15 minutes, 30 minutes, 1 hour)
            if selected_timeframe in ["1m", "5m", "15m", "30m", "60m"]:
                # Use a fixed allowed period for intraday data (e.g., '5d')
                forex_data = yf.download(ticker, period="5d", interval=selected_timeframe)
            elif selected_timeframe == "1y":
                today_date = datetime.today()
                start_date = (today_date - timedelta(days=365)).strftime("%Y-%m-%d")
                end_date = today_date.strftime("%Y-%m-%d")
                forex_data = yf.download(ticker, start=start_date, end=end_date)
            elif selected_timeframe == "5y":
                today_date = datetime.today()
                start_date = (today_date - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
                end_date = today_date.strftime("%Y-%m-%d")
                forex_data = yf.download(ticker, start=start_date, end=end_date)

            # Generate the plot if data is available.
            if forex_data is not None and not forex_data.empty:
                plt.figure(figsize=(10, 6))
                plt.plot(forex_data['Close'], label=f'{selected_pair} Closing Price')
                
                # Adjust date formatting based on timeframe type
                if selected_timeframe in ["1m", "5m", "15m", "30m", "60m"]:
                    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                else:
                    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
                    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                    
                plt.title(f"{selected_pair} Forex - Closing Price ({selected_timeframe})")
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.grid(True)
                plt.legend()
                os.makedirs("static", exist_ok=True)
                # Save plot into the static folder
                plot_path = os.path.join("Flask/static", "plot.png")
                plt.savefig(plot_path)
                plt.close()

    return render_template(
        "chart.html",
        user=current_user,
        currency_pairs=currency_pairs,
        plot_path=plot_path,
        selected_pair=selected_pair,
        selected_timeframe=selected_timeframe
    )



    if request.method == "POST":
        # Get the selected currency pair
        selected_pair = request.form["currency_pair"]

        # Calculate the 5-year date range
        today_date = datetime.today()
        start_date = (today_date - timedelta(days=5 * 365)).strftime("%Y-%m-%d")
        end_date = today_date.strftime("%Y-%m-%d")

        # Fetch Forex data
        forex_data = get_forex_data(currency_pairs[selected_pair], start_date, end_date)

        if forex_data is not None:
            # Generate a plot
            plt.figure(figsize=(10, 6))
            plt.plot(forex_data['Close'], label=f'{selected_pair} Closing Price')
            plt.gca().xaxis.set_major_locator(mdates.YearLocator())
            plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            plt.title(f"{selected_pair} Forex - Closing Price (Last 5 Years)")
            plt.xlabel('Date')
            plt.ylabel('Price')
            plt.grid(True)
            plt.legend()
            os.makedirs("static", exist_ok=True)
            # Save plot
            plot_path = os.path.join("Flask/static", "plot.png")
            plt.savefig(plot_path)
            plt.close()
            
    return render_template(
        "chart.html",
        currency_pairs=currency_pairs,
        plot_path=plot_path,
        selected_pair=selected_pair
    )
   

# @views.route('/dataset/<year>', methods=['GET'])
# def dataset(year):
#     # Initialize page number
#     page = request.args.get('page', 1, type=int)  # Get current page from URL or default to 1
    
#     # Path to the CSV file
#     csv_path = f'static/data/{year}.csv'
    
#     try:
#         # Load the dataset (first 1000 rows)
#         df = pd.read_csv(csv_path)
#         df = df.head(1000)  # Limit to first 1000 rows

#         # Ensure the correct column names
#         df.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close']

#         # Pagination logic
#         per_page = 20  # Number of rows per page
#         total_rows = len(df)
#         total_pages = total_rows // per_page + (1 if total_rows % per_page != 0 else 0)
        
#         start = (page - 1) * per_page
#         end = start + per_page
#         data_page = df.iloc[start:end]  # Slice the DataFrame for the current page

#         # Convert the DataFrame to an HTML table
#         data_html = data_page.to_html(classes='table table-striped', index=False)

#         # Calculate page range for pagination (show pages in groups of 5)
#         start_page = ((page - 1) // 5) * 5 + 1
#         end_page = min(start_page + 4, total_pages)
        
#         previous_page = max(1, page - 5)
#         next_page = min(total_pages, page + 5)

#     except FileNotFoundError:
#         data_html = f"<p>Dataset for {year} not found.</p>"
#         total_pages = 0  # Set total_pages to 0 when file is not found

#     return render_template('dataset.html', year=year, data=data_html, total_pages=total_pages, current_page=page,
#                            start_page=start_page, end_page=end_page, previous_page=previous_page, next_page=next_page)
