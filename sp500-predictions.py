import yfinance as yf
import pandas as pd
import os
if os.path.exists("sp500.csv"):
    sp500 = pd.read_csv("sp500.csv", index_col=0)
else:
    sp500 = yf.Ticker("^GSPC")
    sp500 = sp500.history(period="max")
    sp500.to_csv("sp500.csv")
sp500.index = pd.to_datetime(sp500.index)

# plot closing price against the index
closing_data = sp500.plot(y = "Close", use_index=True)
closing_data.show()