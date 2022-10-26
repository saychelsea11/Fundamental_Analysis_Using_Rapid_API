import streamlit as st
import time
import extract_plot_stock_info_yf97 as ep

st.write("""
## Stock Comparison and Fundamental Analysis
""")

time.sleep(0.5)

#Accepting stock tickers from the user using Streamlit syntax
default_values = "ADI,INTC"
input_tickers = st.text_input("Enter stock tickers separated by commas",default_values)

try:
    stocks = input_tickers.split(',')
    stocks = list(map(str.strip,stocks))
    watchlist = ep.store_stock_info(stocks)
    st.write(watchlist[['Company','Price($)','Market Cap($)','P/S Ratio','Trailing P/E','Forward P/E','PEG Ratio','Price/Book Ratio','Debt/Equity Ratio',
                        'Trailing EPS','Forward EPS','Return On Equity(%)','Return On Assets(%)','Div Yield(%)']])
    ep.plot_dash(watchlist)
    st.write("""
    Enter tickers in the box above to analyze more stocks...
    """)
except Exception as e: 
    st.write("""
    An error occurred. Try entering the tickers again separated by commas.
    """)
    print (str(e))
    pass
