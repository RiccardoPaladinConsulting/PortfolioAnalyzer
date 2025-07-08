# pip install streamlit
#from .config import *
#from .Utils import *
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas_datareader import data
from datetime import date, timedelta
import yfinance as yf
import seaborn as sn
from sklearn.linear_model import LinearRegression
from scipy.stats import kurtosis, skew
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
#import chart
#import altair as alt
from pypfopt import EfficientFrontier
import pandas_datareader as web


def GetHistoricalPrice(tickers, start_date, end_date, price='Close'):
    """
    Fetches historical stock prices using pandas_datareader (Stooq source)
    for given tickers and date range.
    Returns a pandas Series (for single ticker) or DataFrame (for multiple tickers).
    """
    #st.write(f"Attempting to download data for tickers: {tickers} from {start_date} to {end_date} using Stooq...")

    try:
        # pandas_datareader with 'stooq' source can handle a list of tickers
        # It returns a MultiIndex DataFrame if multiple tickers are requested,
        # with the first level being the metric (e.g., 'Close') and the second being the ticker.
        data = web.DataReader(tickers, 'stooq', start=start_date, end=end_date)
        #st.write("Raw data from pandas_datareader (Stooq):")
        #st.write(data)

        if data.empty:
            st.warning(
                "pandas_datareader (Stooq) returned an empty DataFrame. This often means no data is available for the specified range or ticker(s).")
            return pd.DataFrame()  # Return empty DataFrame if no data is fetched

        # Stooq returns data with columns like ('Close', 'AAPL'), ('Close', 'SPY')
        # We need to select the 'Close' prices for all tickers.
        # The data will have a MultiIndex for columns if multiple tickers.
        # Example: data.columns = [('Close', 'AAPL'), ('Volume', 'AAPL'), ...]
        # We want to select all columns where the first level is 'Close'.

        # Check if the columns are MultiIndex (for multiple tickers)
        if isinstance(data.columns, pd.MultiIndex):
            # Select only the 'Close' prices across all tickers
            res = data.loc[:, (price, slice(None))]  # Select all rows, and columns where first level is 'Close'
            res.columns = res.columns.droplevel(0)  # Drop the 'Close' level from column names
        else:
            # For a single ticker, it's a regular DataFrame, so just select the 'Close' column
            res = data[price]
            # Ensure it's a DataFrame for consistent processing later
            if isinstance(res, pd.Series):
                res = res.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)

        # Stooq returns data with the date as the index, but in descending order.
        # We need to sort it to ascending order for ffill/bfill to work correctly.
        res = res.sort_index(ascending=True)

        return res

    except Exception as e:
        st.error(f"Error fetching data from Stooq: {e}")
        st.warning("Please ensure the ticker symbol(s) are valid and you have an internet connection.")
        return pd.DataFrame()
#st.set_option('deprecation.showPyplotGlobalUse', False)
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

st.set_page_config(
    page_title="Stock fundamental analysis",
    page_icon="🧊",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://paladinconsulting.it',
        'Report a bug': "https://paladinconsulting.it",
        'About': "# The web app aims to offer a symple tool to analyze stocks returns. It doesn't constitute any financial advice"
    }
)
st.title('📈 Stock Fundamental Analysis')
st.markdown('## *Author: Riccardo Paladin*')
st.markdown(
    'Unleash your inner financial wizard! '
    'This interactive web app lets you effortlessly dive deep into stock fundamentals and supercharge your portfolio with intelligent, Python-powered optimization.')

st.markdown('📊 Insert your tickers and start the analysis')

col1, col2, col3 = st.columns(3)

with col1:
    # Allow comma-separated tickers and parse them into a list
    ticker_input = st.text_input("Stock Symbols", "SPY,AAPL").upper()
    # Split the input string by comma, strip whitespace, and filter out empty strings
    tickers_list = [t.strip() for t in ticker_input.split(',') if t.strip()]

with col2:
    # Set default start date to 30 days ago
    today = date.today()
    start_date = st.date_input("Start Date", today - timedelta(days=30))

with col3:
    # Set default end date to yesterday to ensure data is available
    end_date = st.date_input("End Date") # Changed to yesterday

# Basic validation for dates
if start_date > end_date:
    st.error("Error: End Date cannot be before Start Date. Please adjust your dates.")
    st.stop() # Stop execution if dates is invalid

tickers_list = [t.strip() for t in ticker_input.split(',') if t.strip()]
st.markdown("""<small>* The first ticker will be used as benchmark</small>""", unsafe_allow_html=True)
col_left, col_center, col_right = st.columns([1, 1, 1])
with col_center:
    fetch_button = st.button("Start Analysis")
if fetch_button:
    if not tickers_list:
        st.warning("Please enter a ticker symbol.")
    else:
        #st.info(f"Fetching data for {tickers_list} from {start_date} to {end_date}...")

        try:
            # 1. Get historical price using your function
            # The yfinance download function returns a DataFrame with a DatetimeIndex
            Stocks_prices = GetHistoricalPrice(tickers_list,start_date,end_date, price='Close')
            # If only one ticker, yfinance might return a Series. Convert to DataFrame if needed.
            if isinstance(Stocks_prices, pd.Series):
                Stocks_prices = Stocks_prices.to_frame(name='Close')

            if Stocks_prices.empty:
                st.warning(f"No historical data found for {tickers_list} in the specified range. Please check the ticker or date range.")
            else:
                # Ensure the index is a DatetimeIndex for reindexing
                Stocks_prices.index = pd.to_datetime(Stocks_prices.index)

                # 2. Generate all weekdays in the range
                all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B') # 'B' for business day frequency

                # 3. Reindex the DataFrame to include all weekdays
                # This will add rows for missing dates (e.g., weekends, holidays)
                Stocks_prices = Stocks_prices.reindex(all_weekdays)

                # 4. Fill forward missing values (e.g., for weekends/holidays)
                # This ensures that non-trading days show the last known price
                Stocks_prices = Stocks_prices.fillna(method='ffill')

                # 5. Fill any remaining NaNs at the beginning if the first date was a non-trading day
                # For example, if start_date is a Saturday and there's no prior Friday data.
                Stocks_prices = Stocks_prices.fillna(method='bfill')

                # Drop any rows that are still NaN (e.g., if the entire range is missing data)
                Stocks_prices.dropna(inplace=True)

                if Stocks_prices.empty:
                    st.warning("After processing, the DataFrame is empty. This might mean no valid data was available for the selected period.")
                else:
                    st.success(f"Data for {tickers_list} from {start_date} to {end_date} processed successfully!")
                    st.subheader("Data Overview")
                    Stocks_prices_dis = Stocks_prices.reset_index()
                    Stocks_prices_dis = Stocks_prices_dis.rename(columns={'index': 'Date'})
                    Stocks_prices_dis['Date'] = pd.to_datetime(Stocks_prices_dis['Date']).dt.date
                    st.dataframe(Stocks_prices_dis) # Display the DataFrame

                    st.markdown("---")


        except Exception as e:
            st.error(f"An error occurred while fetching or processing data: {e}")
            st.warning("Please ensure the ticker symbol is valid and you have an internet connection.")


##CHART
try:
    if not Stocks_prices.empty:
        st.markdown('##  Charts')
        st.subheader("Daily Price")
        fig_prices, ax_prices = plt.subplots(figsize=(10, 6))
        Stocks_prices.plot(ax=ax_prices)
        ax_prices.set_title(f"Historical Closing Prices for {', '.join(tickers_list)}")
        ax_prices.set_xlabel("Date")
        ax_prices.set_ylabel("Price")
        ax_prices.grid(True)
        st.pyplot(fig_prices) # Display the plot in Streamlit
        plt.close(fig_prices) # Close the figure to free up memory

        st.subheader("Daily Percentage Change")
                    # Calculate percentage change and drop NaNs
        Stocks_pct_change = Stocks_prices.pct_change().dropna()

        if Stocks_pct_change.empty:
            st.warning("No sufficient data to calculate percentage change.")
        else:
            fig_pct_change, ax_pct_change = plt.subplots(figsize=(10, 6))
            Stocks_pct_change.plot(ax=ax_pct_change)
            ax_pct_change.set_title(f"Daily Percentage Change for {', '.join(tickers_list)}")
            ax_pct_change.set_xlabel("Date")
            ax_pct_change.set_ylabel("Percentage Change")
            ax_pct_change.axhline(0, color='grey', linestyle='--', linewidth=0.8) # Add a horizontal line at
            ax_pct_change.grid(True)
            st.pyplot(fig_pct_change) # Display the plot in Streamlit
            plt.close(fig_pct_change) # Close the figure to free up memory
except Exception as e:
    print('')
###

try:
    if not Stocks_pct_change.empty:

        st.markdown("---")
        st.markdown('##  Fundamental analysis')

        fundamentals = []
        mean_ret = [] # This variable is not used after calculation, can be removed if not needed elsewhere


                                # Iterate over each stock's percentage change data
        for (columnName, columnData) in Stocks_pct_change.items():
                                    # Annualize mean return (assuming 252 trading days in a year)
            means = columnData.mean() * 252
                                    # Annualize standard deviation (volatility)
            stds = columnData.std() * (252 ** 0.5)
                                    # Calculate Sharpe Ratio (assuming risk-free rate is 0 for simplicity here)
            sharpe = means / stds if stds != 0 else 0 # Avoid division by zero

            fundamentals.append(
                {'Stock': columnName,
                 'Annualized Mean Return': f"{means*100:.2f}", # Format to 4 decimal places
                 'Annualized Standard Deviation': f"{stds*100:.2f}", # Format to 4 decimal places
                 'Sharpe Ratio': f"{sharpe:.4f}" # Format to 4 decimal places
                 }
            )
        #
        reg_data = []
        for i in range(len(Stocks_prices.columns)):
            model = LinearRegression()
            X = Stocks_prices.iloc[0:, 0].to_numpy().reshape(-1, 1)
            Y = Stocks_prices.iloc[0:, i].to_numpy().reshape(-1, 1)
            reg = model.fit(X, Y)
            alpha = float(reg.intercept_)
            beta = float(reg.coef_)
            reg_data.append(
                {'Alpha': f"{alpha:.3f}",
                 'Beta': f"{beta:.2f}"

                 })
        fundamentals = pd.DataFrame(fundamentals)
        reg_data = pd.DataFrame(reg_data)
        fundamentals = fundamentals.join(reg_data)
        fundamentals = fundamentals.sort_values(by='Sharpe Ratio', ascending=False)

        st.dataframe(fundamentals) #Show dataframe
        st.markdown("---")
        st.markdown('## Correlation matrix')
        fig, ax = plt.subplots()
        sns.heatmap(Stocks_prices.corr(), ax=ax)
        corr = st.write(fig)


        st.markdown(
            f"""
            {corr}
            """
        )

except Exception as e:
    print('')



try:
    if not Stocks_prices.empty:

        st.markdown('## Portfolio optimization ')
        Portfolio_selected = Stocks_prices
        p_ret = []
        p_vol = []
        p_weights = []
        num_assets = len(Portfolio_selected.columns)
        num_portfolios = 10000
        cov_matrix = Portfolio_selected.apply(lambda x: np.log(1 + x)).cov()

        mean_returns_annual = []
        for (columnName, columnData) in Portfolio_selected.items():
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
        st.markdown('Weights for the minimum variance portfolio ')
        st.dataframe(min_vol_port)#Show dataframe




        optimal_risky_port = portfolios_generated.iloc[((portfolios_generated['Returns']) /
                                                        portfolios_generated['Volatility']).idxmax()]
        st.markdown('Weights for the maximum Sharpe Ratio  portfolio ')
        st.dataframe(optimal_risky_port)#Show dataframe





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

except Exception as e:
    print('')

try:
    if not Stocks_prices.empty:
        st.markdown('## Predictions next 20 days ')
        prediction = []
        MSE = []
        for i in range(len(Stocks_prices.columns)):
            model = LinearRegression()
            model.fit(Stocks_prices.iloc[0:len(Stocks_prices) - 20, [-i]], Stocks_prices.iloc[0:len(Stocks_prices) - 20, i])
            pred = model.predict(Stocks_prices.iloc[len(Stocks_prices) - 20:, [-i]])
            prediction.append(pred)
            mse = np.sqrt(mean_squared_error(Stocks_prices.iloc[len(Stocks_prices) - 20:, i], pred))
            MSE.append(mse)

        prediction = np.asarray(prediction)
        prediction = prediction.tolist()
        df = pd.DataFrame(prediction).T
        df.columns = list(df.columns)
        Stocks1 = pd.concat([Stocks_prices, df], ignore_index=True)
        st.dataframe(Stocks1) #Show dataframe

except Exception as e:
    print('')

try:

    MSE_mean = sum(MSE) / len(MSE)
    st.markdown('Mean Squared Error of the Predictions ')
    st.markdown(
        f"""
        {MSE_mean}
        """
    )
except Exception as e:
    print('')

