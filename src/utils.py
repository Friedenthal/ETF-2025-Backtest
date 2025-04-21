import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import datetime as dt
import itertools as it
import ast


"""
Excel_Date() converts a datetime object to an Excel date format

revert_Excel_Date() converts an Excel date format back to a datetime object
"""
def Excel_Date(date1):
    temp = dt.datetime(1899,12,30)
    delta = date1 - temp
    return float(delta.days)
def revert_Excel_Date(excel_date):
    # Excel dates start from December 30, 1899
    temp = dt.datetime(1899, 12, 30)
    # Convert the Excel date back to a datetime object
    reverted_date = temp + dt.timedelta(days=excel_date)
# Convert to pandas Timestamp
    pandas_date = pd.Timestamp(reverted_date)
# Format as YYYY-MM-DD
    formatted_date = pandas_date.strftime('%Y-%m-%d')
    return formatted_date

"""
performance_summary(df_performance, days, start_date, end_date) compares the performance summary of various portfolios, when each column in df_performance is the % changes of a portfolio; days --> frequency of data in number of trading days
"""

def performance_summary(df_performance, days, start_date = None, end_date = None):
    df_stats = []
    
    # Adjust for start_date and end_date if provided
    if start_date is not None and end_date is not None:
        df_performance = df_performance.loc[start_date:end_date]
    elif start_date is not None:
        df_performance = df_performance.loc[start_date:]
    elif end_date is not None:
        df_performance = df_performance.loc[:end_date]

    interval_days = df_performance.index[-1] - df_performance.index[0]
    for columns in df_performance.columns:
        portfolio = (1 + df_performance[columns]).cumprod()
        annualized_return = (portfolio.iloc[-1]) ** (365 / interval_days.days) - 1
        standard_deviation = df_performance[columns].std() * np.sqrt(252/days)
        less_than_zero = df_performance[columns][df_performance[columns] < 0]
        downside_deviation = less_than_zero.std() * np.sqrt(252/days)
        df_drawdown = ((portfolio - portfolio.cummax())/portfolio.cummax())
        max_drawdown = df_drawdown.min()
        df_stats.append({"Group": columns, "Annualized Return": round(annualized_return*100, 2), "Standard Deviation": round(100*standard_deviation, 2), "Downside Deviation": round(100*downside_deviation, 2), "Max Drawdown": round(100*max_drawdown, 2)})
    df_stats = pd.DataFrame(df_stats)

    return df_stats

"""
Performance Summary Benchmarking function compares the Performance Summary of the Portfolio with the Universe Mean Benchmark & an S&P 500 fund benchmark (VOO US Equity)
"""
def backtest_benchmarks(df_performance, df_price, days, start_date = None, end_date = None):
    df_pct_change = df_price.sort_index(ascending = True).pct_change().dropna(how = "all").fillna(0)
    df_stats = performance_summary(df_performance, days, start_date, end_date)
    df_pct_change["Universe Mean"] = df_pct_change.mean(axis = 1)
    df_pct_change["S&P 500"] = df_pct_change["VOO US Equity"]
    df_pct_change.index = df_pct_change.index.map(lambda x: revert_Excel_Date(x))
    df_pct_change.index = pd.to_datetime(df_pct_change.index, format='%Y-%m-%d')
    df_benchmark_stats = performance_summary(df_pct_change[["Universe Mean", "S&P 500"]], days, start_date, end_date)
    df_benchmark_stats.set_index("Group", inplace = True)
    df_stats.set_index("Group", inplace = True)
    df_stats.loc["Universe Mean"] = df_benchmark_stats.loc["Universe Mean"]
    df_stats.loc["S&P 500"] = df_benchmark_stats.loc["S&P 500"]

    if start_date is None:
        print("Start Date: ", df_performance.index[0])
    else:
        print("Start Date: ", start_date)
    if end_date is None:
        print("End Date: ", df_performance.index[-1])
    else:
        print("End Date: ", end_date)
    
    return df_stats

"""
Regression_Gen(df_price) generates the Score_Summary using the regression_array parameters. "Cheapness" & "Combined Term 2" have been averaged & the combined score is generated as given in ./Raw Data/raw_data.txt
df_price_subset must be in descending order of dates, with the index being the dates in Excel format
"""

"""
regression_array (Old) --> list(range(10, 126, 5))

regression_array (Current) --> [25, 30, 40, 55, 75, 100, 130, 165, 205, 250]
"""
def Regression_Gen(df_price_subset, regression_array):
    #df_ind_scores = pd.DataFrame(index = df_price_subset.columns, columns = ["Slope", "Cheapness"])
    ticker_summary = []
    x = df_price_subset.index
    for days_shift in range(0, len(df_price_subset.index) - 250):
    #for days_shift in range(0, 10):
        for ticker in df_price_subset.columns:
            y = df_price_subset[ticker].values
            #print(x)
            #print(y)
            Score_slope = 0
            Score_cheapness = 0
            RSQ_reg_price = 0
            RSQ_nom_slope = 0
            for i in regression_array:
                Linear_Regression = stats.linregress(x[days_shift:i+days_shift], y[days_shift:i+days_shift])
                slope = Linear_Regression.slope
                intercept = Linear_Regression.intercept
                cur_price = y[days_shift]
                reg_price = (slope * x[days_shift] + intercept)
                r_squared = Linear_Regression.rvalue ** 2
                #old_score = old_score +  r_squared * ((slope * (x[days_shift] + 32) + intercept) - cur_price) / cur_price
                Score_slope = Score_slope + (slope/reg_price) * r_squared
                Score_cheapness = Score_cheapness + r_squared * (reg_price - cur_price) / cur_price

                # RSQ weighted Reg Price
                RSQ_reg_price = RSQ_reg_price + r_squared * reg_price

                # RSQ weightded nominal slope
                RSQ_nom_slope = RSQ_nom_slope + r_squared * slope
            ticker_summary.append({"Date":x[days_shift], "Ticker": ticker, "Slope": Score_slope, "Cheapness": Score_cheapness, "RSQ weighted Reg Price": RSQ_reg_price, "RSQ weighted Nominal Slope": RSQ_nom_slope})
    df_score_summary = pd.DataFrame(ticker_summary)

    df_score_summary["Cur Price"] = df_score_summary.apply(lambda x: df_price_subset.loc[x["Date"], x["Ticker"]], axis = 1)
    df_score_summary["Cheapness"] = df_score_summary["Cheapness"] / len(regression_array)
    df_score_summary["Combined Term 2"] = df_score_summary["RSQ weighted Nominal Slope"] / df_score_summary["Cur Price"]
    df_score_summary["Combined Term 2"] = df_score_summary["Combined Term 2"] / len(regression_array)
    return df_score_summary

"""
SwapList_Generator() arguments --> scorefile, date, tickers_held, swap_threshold
Swaplist_Generator() generates the list of tickers to be swapped out of the portfolio based on the swap_threshold. The function returns a dataframe with the tickers to be swapped out and their respective scores.
"""

def SwapList_Generator(df_scores, cur_date, tickers_held, swap_threshold):
    all_tickers = df_scores.loc[cur_date].dropna().index
    available_tickers = list(set(all_tickers) - set(tickers_held))
    #print(pd.DataFrame(df_scores.loc[cur_date][tickers_held]))
    df_held_scores = pd.DataFrame(df_scores.loc[cur_date][tickers_held]).sort_values(by = cur_date, ascending = True)
    df_available_scores = pd.DataFrame(df_scores.loc[cur_date][available_tickers]).sort_values(by = cur_date, ascending = False)
    df_max_swaps = df_available_scores.iloc[:len(tickers_held)]
    df_max_swaps.reset_index(inplace = True)
    df_held_scores.reset_index(inplace = True)
    #print(df_max_swaps)
    #print(df_held_scores)
    df_max_swaps["Existing Ticker"] = df_held_scores["Ticker"]
    df_max_swaps["Current Score"] = df_held_scores[cur_date]
    #print(df_max_swaps)
    df_max_swaps.set_index("Ticker", inplace = True)
    df_max_swaps["Score Difference"] = df_max_swaps[cur_date] - df_max_swaps["Current Score"]
    #print(df_max_swaps)
    df_swaps = df_max_swaps[df_max_swaps["Score Difference"] >= swap_threshold]
    return df_swaps

"""
Portfolio_Tracker tracks the portfolio value over time. It takes the following arguments:
df_scores --> dataframe of scores for each ticker
df_price --> dataframe of prices for each ticker
date_range --> date range for the portfolio tracker
num_positions --> number of ETFs to be held in the portfolio
swap_threshold --> threshold for swapping out tickers
trade_threshold --> threshold for a trade day based on max_score_difference
initial_capital --> initial capital for the portfolio
trade_cost_assumption --> trade cost assumption for each swap......default is 10bps
"""
def Portfolio_Tracker(df_scores, df_price, date_range, num_positions, swap_threshold, trade_threshold, initial_capital = 1000, trade_cost_assumption = 0.001):
    # df_price must be in ascending order by date to get the accurate pct_change
    df_pct_change = df_price.sort_index(ascending = True).pct_change().dropna(how = "all").fillna(0)
    df_holdings = []
    tickers_held = df_scores.loc[date_range[0]].nlargest(num_positions).index
    df_weights = [1/num_positions] * num_positions
    portval = initial_capital
    Trade_tracker = True
    tickers_swapped_count = 0
    for index, date in enumerate(date_range[1:]):
        tickers_held = sorted(tickers_held)
        pct_changes = df_pct_change.loc[date, tickers_held]
        df_weights = [weight * (1 + pct_change) for weight, pct_change in zip(df_weights, pct_changes)]
        sum_weights = sum(df_weights)
        df_weights = [weight/sum_weights for weight in df_weights]
        portval = sum_weights * portval
        swaplist = SwapList_Generator(df_scores, date, tickers_held, swap_threshold)
        #print(date)
        if swaplist is not None:
            max_score_diff = swaplist["Score Difference"].max()
            if max_score_diff >= trade_threshold:
                tickers_swapped = swaplist["Existing Ticker"].tolist()
                #print(tickers_swapped)
                tickers_held = list(set(tickers_held) - set(tickers_swapped))
                #print(tickers_held)
                tickers_added = swaplist.index.tolist()
                #print(tickers_added)
                tickers_held = list(set(tickers_held) | set(tickers_added))
                #print(tickers_held)
                tickers_held = sorted(tickers_held)
                df_weights = [1/num_positions] * num_positions
                Trade_tracker = True
                tickers_swapped_count = len(tickers_swapped)
        df_holdings.append({"Date": date, "Tickers": tickers_held, "Portfolio Value": portval, "Weights": df_weights, "Trade Tracker": Trade_tracker, "Number of Swaps": tickers_swapped_count})
        Trade_tracker = False
        tickers_swapped_count = 0
    df_holdings = pd.DataFrame(df_holdings)
    df_holdings.set_index("Date", inplace = True)
    df_holdings["Return %"] = df_holdings["Portfolio Value"].pct_change()
    df_holdings["Return % adjusted for swaps"] = (df_holdings["Return %"] - df_holdings["Number of Swaps"] * trade_cost_assumption / num_positions)

    df_holdings.index = df_holdings.index.map(lambda x: revert_Excel_Date(x))
    df_holdings.index = pd.to_datetime(df_holdings.index, format='%Y-%m-%d')
    return df_holdings