import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from datetime import datetime, timedelta

# # download data for the past 5 years
# symbol = 'EURUSD=X'
# end_date = datetime.now().strftime('%Y-%m-%d')
# start_date = (datetime.now() - timedelta(days=5*365)).strftime('%Y-%m-%d')
# data = yf.download(symbol, start=start_date, end=end_date, progress=False)
# data = data[['Close']]

# compute the MACD
exp12 = data['Close'].ewm(span=12, adjust=False).mean()
exp26 = data['Close'].ewm(span=26, adjust=False).mean()
macd_line = exp12 - exp26
signal_line = macd_line.ewm(span=9, adjust=False).mean()
data['MACD_hist'] = macd_line - signal_line

# compute RSI using a 14-day period
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
data['RSI'] = 100 - (100 / (1 + rs))

# compute the Stochastic Oscillator over a 14-day window
low14 = data['Close'].rolling(14).min()
high14 = data['Close'].rolling(14).max()
data['%K'] = (data['Close'] - low14) / (high14 - low14) * 100

# create fuzzy variables
macd_range = np.arange(-1, 1, 0.001)
rsi_range = np.arange(0, 100, 0.1)
stochastic_range = np.arange(0, 100, 0.1)
action_range = np.arange(-1, 1.01, 0.01)

macd = ctrl.Antecedent(macd_range, 'macd')
rsi = ctrl.Antecedent(rsi_range, 'rsi')
stochastic = ctrl.Antecedent(stochastic_range, 'stochastic')
action = ctrl.Consequent(action_range, 'action')

# define membership functions for MACD, RSI, SO and the trading action
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

# define fuzzy rules
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

# create a control system
trading_ctrl = ctrl.ControlSystem(rules)
trading_sim = ctrl.ControlSystemSimulation(trading_ctrl)

# create signals and action values
signals = []
action_values = []
for idx in range(len(data)):
    try:
        trading_sim.input['macd'] = data['MACD_hist'].iloc[idx]
        trading_sim.input['rsi'] = data['RSI'].iloc[idx]
        trading_sim.input['stochastic'] = data['%K'].iloc[idx]
        trading_sim.compute()
        action_value = trading_sim.output['action']
        
        # trading strategy intervals
        if action_value <= -0.25:
            signals.append('sell') # sell if action_value <= -0.25
        elif -0.25 < action_value <= 0.25:
            signals.append('hold') # hold if in the hold range
        else:  
            signals.append('buy') # buy if action_value > 0.25
            
        action_values.append(action_value)
    except:
        signals.append('hold')
        action_values.append(0)

data['Signal'] = signals
data['ActionValue'] = action_values

# handle any missing values in the data
data = data.dropna()

# calculate returns and strategy performance
data['Returns'] = data['Close'].pct_change().fillna(0)

# initialize position and strategy columns
data['Position'] = 0
data.loc[data['Signal'] == 'buy', 'Position'] = 1
data.loc[data['Signal'] == 'sell', 'Position'] = -1

# calculate strategy returns
data['Strategy Returns'] = data['Position'].shift(1) * data['Returns']
data['Strategy Returns'] = data['Strategy Returns'].fillna(0)

# calculate cumulative returns starting from 1
data['CumMarket'] = (1 + data['Returns']).cumprod()
data['CumStrategy'] = (1 + data['Strategy Returns']).cumprod()

# print strategy performance metrics with error handling
try:
    total_return_strategy = (data['CumStrategy'].iloc[-1] - 1) * 100
    total_return_market = (data['CumMarket'].iloc[-1] - 1) * 100
    
    # output the performance metrics
    print(f"\nStrategy Performance Summary for {symbol}:")
    print(f"Total Strategy Return: {total_return_strategy:.2f}%")
    print(f"Total Market Return: {total_return_market:.2f}%")
    print(f"Number of Buy Signals: {len(data[data['Signal'] == 'buy'])}")
    print(f"Number of Sell Signals: {len(data[data['Signal'] == 'sell'])}")
except Exception as e:
    print(f"Error calculating returns: {e}")

# Data Visualization
# Figure 1: Technical Analysis with Trading Signals
plt.figure(figsize=(16, 12))

# price and trading signals
ax1 = plt.subplot(4, 1, 1)
plt.plot(data['Close'], label='Price', color='blue')
plt.title(f'Technical Analysis with Trading Signals - {symbol}')

# plot buy/sell signals
buy_signals = data[data['Signal'] == 'buy']
sell_signals = data[data['Signal'] == 'sell']
plt.scatter(buy_signals.index, buy_signals['Close'], c='g', marker='^', s=50, label='Buy')
plt.scatter(sell_signals.index, sell_signals['Close'], c='r', marker='v', s=50, label='Sell')
plt.legend()
plt.grid(True)

# MACD
ax2 = plt.subplot(4, 1, 2, sharex=ax1)
plt.plot(data['MACD_hist'], label='MACD Histogram', color='blue')
plt.axhline(y=0, color='r', linestyle='--')
plt.legend()
plt.grid(True)
plt.title('MACD Histogram')

# RSI
ax3 = plt.subplot(4, 1, 3, sharex=ax1)
plt.plot(data['RSI'], label='RSI', color='purple')
plt.axhline(y=70, color='r', linestyle='--')
plt.axhline(y=30, color='g', linestyle='--')
plt.legend()
plt.grid(True)
plt.title('RSI')

# Stochastic Oscillator
ax4 = plt.subplot(4, 1, 4, sharex=ax1)
plt.plot(data['%K'], label='%K', color='orange')
plt.axhline(y=80, color='r', linestyle='--')
plt.axhline(y=20, color='g', linestyle='--')
plt.legend()
plt.grid(True)
plt.title('Stochastic Oscillator')
plt.tight_layout()
plt.show()

# Figure 2: Cumulative Returns Comparison
plt.figure(figsize=(16, 12))

ax1 = plt.subplot(2, 1, 1)
plt.plot(data['Close'], label='Price')
buy_signals = data[data['Signal'] == 'buy']
sell_signals = data[data['Signal'] == 'sell']

plt.scatter(buy_signals.index, buy_signals['Close'], c='g', marker='^', s=50, label='Buy')
plt.scatter(sell_signals.index, sell_signals['Close'], c='r', marker='v', s=50, label='Sell')
plt.title(f'Trading Signals for {symbol}')
plt.legend()

ax2 = plt.subplot(2, 1, 2, sharex=ax1)
plt.plot(data['CumMarket'], label='Market')
plt.plot(data['CumStrategy'], label='Strategy')
plt.title(f'Cumulative Returns Comparison - {symbol}')
plt.legend()
plt.tight_layout()
plt.show()