# pip install streamlit
from .config import *
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
from datetime import date
import yfinance as yf
import seaborn as sn
from sklearn.linear_model import LinearRegression
from scipy.stats import kurtosis, skew
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import chart
import altair as alt
from pypfopt import EfficientFrontier

st.set_option('deprecation.showPyplotGlobalUse', False)

st.set_page_config(
    page_title="Stock fundamental analysis")

st.title('📈 Stock Fundamental Analysis')
st.markdown('## **Authors: Riccardo Paladin, Gabriella Saade, Nhat Pham**')
st.markdown(
    'In this web app you can insert stock tickers and obtain a complete fundamental analysis and portfolio optimization.'
    'It is based on machine learning algorithms implemented in python.')

st.markdown('📊 Insert a series of tickers and start the analysis')

tickers_input = st.text_input(' 📍 Enter here the tickers and in the first position the benchmark (no commas)',
                              '').split()

start_date = st.text_input('🗓 Enter here the start date (mm-dd-yyyy)', '')
end_date = st.text_input('🗓 Enter here the end date (mm-dd-yyyy)', '')

Data = data.DataReader(tickers_input, 'yahoo', start_date, end_date)
Stocks_prices = Data['Adj Close']
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
Stocks_prices = Stocks_prices.reindex(all_weekdays)
Stocks_prices = Stocks_prices.fillna(method='ffill')
st.dataframe(Stocks_prices) #Show dataframe



Stocks = Stocks_prices.pct_change()
Stocks = Stocks.dropna()
Stocks_prices.plot()
plt.show()
st.pyplot()

st.markdown('##  Fundamental analysis')

fundamentals = []
mean_ret = []
for (columnName, columnData) in Stocks.iteritems():
    means = columnData.mean() * 252
    mean_ret.append(means)
    stds = columnData.std() * (252 ** 0.5)
    sharpe = means / stds
    fundamentals.append(
        {'Stock': columnName,
         'Mean': means,
         'Standard Dev': stds,
         'Sharpe': sharpe

         }
    )

reg_data = []
for i in range(len(Stocks.columns)):
    model = LinearRegression()
    X = Stocks.iloc[0:, 0].to_numpy().reshape(-1, 1)
    Y = Stocks.iloc[0:, i].to_numpy().reshape(-1, 1)
    reg = model.fit(X, Y)
    alpha = float(reg.intercept_)
    beta = float(reg.coef_)
    reg_data.append(
        {'Alpha': alpha,
         'Beta': beta

         })
fundamentals = pd.DataFrame(fundamentals)
reg_data = pd.DataFrame(reg_data)
fundamentals = fundamentals.join(reg_data)
fundamentals = fundamentals.sort_values(by='Sharpe', ascending=False)


def downside_risk(rets, risk_free=0):
    adj_returns = rets - risk_free
    sqr_downside = np.square(np.clip(adj_returns, np.NINF, 0))
    return np.sqrt(np.nanmean(sqr_downside))


def sortino(rets, risk_free=0):
    adj_returns = rets - risk_free
    drisk = downside_risk(adj_returns)

    if drisk == 0:
        return np.nan

    return (np.nanmean(adj_returns)) / drisk


def get_kurtosis(rets):
    rets1 = rets.to_numpy()
    kurt = kurtosis(rets1, fisher=True)

    return kurt[0]


def get_skew(rets):
    rets1 = rets.to_numpy()
    skewness = skew(rets1)

    return skewness[0]


def get_maximum_drawdown(daily_return_series):
    cum_ret = (daily_return_series + 1).cumprod()
    running_max = np.maximum.accumulate(cum_ret)

    # Ensure the value never drops below 1
    running_max[running_max < 1] = 1

    # Calculate the percentage drawdown
    drawdown = (cum_ret) / running_max - 1

    return drawdown.min()


st.dataframe(fundamentals) #Show dataframe

st.markdown('## Correlation matrix')
fig, ax = plt.subplots()
sns.heatmap(Stocks.corr(), ax=ax)
corr = st.write(fig)


st.markdown(
    f"""
    {corr}
    """
)

st.markdown('## Portfolio optimization ')

Portfolio_selected = Stocks
p_ret = []
p_vol = []
p_weights = []
num_assets = len(Portfolio_selected.columns)
num_portfolios = 10000
cov_matrix = Portfolio_selected.apply(lambda x: np.log(1 + x)).cov()

mean_returns_annual = []
for (columnName, columnData) in Portfolio_selected.iteritems():
    means_a = columnData.mean() * 252
    mean_returns_annual.append(means_a)

for portfolio in range(num_portfolios):
    weights = np.random.uniform(0.05, 0.15, num_assets)
    weights = weights / np.sum(weights)
    p_weights.append(weights)
    returns = np.dot(weights, mean_returns_annual)
    p_ret.append(returns)
    var = cov_matrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()  # Portfolio Variance
    sd = np.sqrt(var)  # Daily standard deviation
    ann_sd = sd * np.sqrt(252)  # Annual standard deviation = volatility
    p_vol.append(ann_sd)

data = {'Returns': p_ret, 'Volatility': p_vol}

for counter, symbol in enumerate(Portfolio_selected.columns.tolist()):
    # print(counter, symbol)
    data[symbol] = [w[counter] for w in p_weights]

portfolios_generated = pd.DataFrame(data)

min_vol_port = portfolios_generated.iloc[portfolios_generated['Volatility'].idxmin()]

st.dataframe(min_vol_port)#Show dataframe


st.markdown('Weights for the minimum variance portfolio ')

optimal_risky_port = portfolios_generated.iloc[((portfolios_generated['Returns']) /
                                                portfolios_generated['Volatility']).idxmax()]

st.dataframe(optimal_risky_port)#Show dataframe

st.markdown('Weights for the maximum Sharpe Ratio  portfolio ')



st.markdown('## Performance for Optimal Portfolio')

Ret = Stocks_prices.pct_change().dropna()
opt_rets = Ret.mean() * 252
opt_cov = Ret.cov() * 252
op = EfficientFrontier(opt_rets, opt_cov, weight_bounds=(0, 1))
w = op.min_volatility()
w1 = op.clean_weights()
opt_w = pd.DataFrame(w1, columns=w1.keys(), index=[0])
opt_w = opt_w.transpose()
port_rets = Ret.dot(opt_w)

port_rets = pd.DataFrame(p_ret)
port_rets.columns = ['Portfolio Returns']
cumrets = np.cumsum(port_rets)  # Cumulative returns
annuals = port_rets.resample('1Y').sum()  # Annulized

sortino_ratio = sortino(annuals, risk_free=0)
kurtosis1 = get_kurtosis(annuals)
skewness1 = get_skew(annuals)
mdd = get_maximum_drawdown(port_rets)
avg_arets = port_rets.mean() * 252
avg_avol = port_rets.std() * 252

performance = pd.DataFrame(np.zeros((6, 1)))
performance.columns = ['Optimal Portfolio']

performance.iloc[0, 0] = avg_arets
performance.iloc[1, 0] = avg_avol
performance.iloc[2, 0] = sortino_ratio
performance.iloc[3, 0] = mdd
performance.iloc[4, 0] = kurtosis1
performance.iloc[5, 0] = skewness1

performance.index = ['Average Returns', 'Average Volatility', 'Sortino Ratio', 'Max. Drawdown', 'Kurtosis', 'Skewness']

st.dataframe(performance) #Show dataframe


st.markdown('## Predictions next 20 days ')

prediction = []
MSE = []
for i in range(len(Stocks.columns)):
    model = LinearRegression()
    model.fit(Stocks.iloc[0:len(Stocks) - 20, [-i]], Stocks.iloc[0:len(Stocks) - 20, i])
    pred = model.predict(Stocks.iloc[len(Stocks) - 20:, [-i]])
    prediction.append(pred)
    mse = np.sqrt(mean_squared_error(Stocks.iloc[len(Stocks) - 20:, i], pred))
    MSE.append(mse)

prediction = np.asarray(prediction)
prediction = prediction.tolist()
df = pd.DataFrame(prediction).T
df.columns = list(Stocks.columns)
Stocks1 = Stocks.append(df, ignore_index=True)
st.dataframe(Stocks1) #Show dataframe



st.markdown('Mean Squared Error of the Predictions ')
MSE_mean = sum(MSE) / len(MSE)
st.markdown(
    f"""
    {MSE_mean}
    """
)
