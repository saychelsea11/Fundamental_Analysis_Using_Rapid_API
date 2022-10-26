import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
import seaborn as sns
import json
import pprint
import streamlit as st
import time

def get_stock_info(ticker):
  url = "https://yahoo-finance97.p.rapidapi.com/stock-info"

  payload = "symbol=" + ticker
  headers = {
    "content-type": "application/x-www-form-urlencoded",
    "X-RapidAPI-Key": "YOUR API KEY",
    "X-RapidAPI-Host": "yahoo-finance97.p.rapidapi.com"
  }

  response = requests.request("POST", url, data=payload, headers=headers)
  data = json.loads(response.text)
  
  return (data, response.status_code)
  
def get_earnings(ticker):
  url = "https://yahoo-finance97.p.rapidapi.com/earnings"
  
  payload = "symbol=" + ticker
  headers = {
    "content-type": "application/x-www-form-urlencoded",
    "X-RapidAPI-Key": "YOUR API KEY",
    "X-RapidAPI-Host": "yahoo-finance97.p.rapidapi.com"
  }

  response = requests.request("POST", url, data=payload, headers=headers)
  data = json.loads(response.text)

  return (data, response.status_code)

def get_financials(ticker):
  url = "https://yahoo-finance97.p.rapidapi.com/financials"

  payload = "symbol=" + ticker
  headers = {
    "content-type": "application/x-www-form-urlencoded",
    "X-RapidAPI-Key": "YOUR API KEY",
    "X-RapidAPI-Host": "yahoo-finance97.p.rapidapi.com"
  }

  response = requests.request("POST", url, data=payload, headers=headers)
  data = json.loads(response.text)

  return (data, response.status_code)

def get_calendar(ticker):
  url = "https://yahoo-finance97.p.rapidapi.com/calendar"
    
  payload = "symbol=" + ticker
  headers = {
    "content-type": "application/x-www-form-urlencoded",
    "X-RapidAPI-Key": "YOUR API KEY",
    "X-RapidAPI-Host": "yahoo-finance97.p.rapidapi.com"
  }

  response = requests.request("POST", url, data=payload, headers=headers)
  data = json.loads(response.text)

  return (data, response.status_code)

def get_quarterly_earnings(ticker):
  url = "https://yahoo-finance97.p.rapidapi.com/quarterly-earnings"

  payload = "symbol=" + ticker
  headers = {
    "content-type": "application/x-www-form-urlencoded",
    "X-RapidAPI-Key": "YOUR API KEY",
    "X-RapidAPI-Host": "yahoo-finance97.p.rapidapi.com"
  }

  response = requests.request("POST", url, data=payload, headers=headers)
  data = json.loads(response.text)

  return (data, response.status_code)

def store_stock_info(tickers):
  company_name = []
  price = []
  market_cap = []
  ps_ratio = []
  trailing_pe = []
  forward_pe = []
  rev_year1 = []
  rev_year2 = []
  rev_year3 = []
  rev_year4 = []
  rev_quarter1 = []
  rev_quarter2 = []
  rev_quarter3 = []
  rev_quarter4 = []
  earnings_year1 = []
  earnings_year2 = []
  earnings_year3 = []
  earnings_year4 = []
  earnings_quarter1 = []
  earnings_quarter2 = []
  earnings_quarter3 = []
  earnings_quarter4 = []
  profit_margin = []
  cash = []
  debt = []
  op_cash = []
  free_cash = []
  ebitda = []
  ps = []
  current_quarter = []
  prev_quarter = []
  next_quarter = []
  earnings_date = []
  div_yield = []
  op_margin = []
  peg_ratio = []
  debt_equity_ratio = []
  price_book_ratio = []
  trailing_eps = [] 
  forward_eps = []
  return_on_equity = []
  return_on_assets = []

  missing_ticker = []
  bad_response = []
  count = 0

  df = pd.DataFrame()

  for tick in tickers:
    print ()
    print("Getting",tick,"info...")
    
    start_time = time.time()
    #data, response_status = extract_stock_info(tick)
    #Extracting eneral stock info
    stock_info_data,stock_info_code = get_stock_info(tick)
    
    time.sleep(0.1)
    
    #Extracting earnings data
    earnings_data,earnings_code = get_earnings(tick)
    
    time.sleep(0.1)
    
    #Extracting quarterly earnings data
    #quarterly_earnings_data,quarterly_earnings_code = get_quarterly_earnings(tick)
    
    #time.sleep(0.1)
    
    print ("Done!")
    stop_time = time.time()
    duration = stop_time - start_time
    print ("API transaction time for",tick,":",duration)
        
    #Parsing stock info metrics
    if stock_info_code == 200: 
      
      #Parsing company name
      company_name.append(stock_info_data['data']['longName'])
      
      #Parsing current stock price
      price.append(np.round(stock_info_data['data']['currentPrice'],2))
      
      #Parsing market cap
      market_cap.append(str(np.round(stock_info_data['data']['marketCap']/1000000000,2)) + 'B')
      
      #Parsing P/S ratio
      try:
        ps_ratio.append(np.round(stock_info_data['data']['priceToSalesTrailing12Months'],2))
      except: 
        ps_ratio.append(0)
      
      #Parsing trailing P/E
      try:
        trailing_pe.append(np.round(stock_info_data['data']['trailingPE'],2))
      except: 
        trailing_pe.append(np.nan)
      
      #Parsing forward P/E
      try:
        forward_pe.append(np.round(stock_info_data['data']['forwardPE'],2))
      except: 
        forward_pe.append(np.nan)
      
      #Parsing profit margin
      profit_margin.append(np.round(stock_info_data['data']['profitMargins']*100,2))
      
      #Parsing cash on hand
      try:
        if stock_info_data['data']['country'] == 'China':
          cash.append(int(stock_info_data['data']['totalCash']/1000000000)*0.16)
        elif stock_info_data['data']['country'] == 'Taiwan':
          cash.append(int(stock_info_data['data']['totalCash']/1000000000)*0.031)
        else: 
          cash.append(int(stock_info_data['data']['totalCash']/1000000000))
      except:
        cash.append(0)
        
      #Parsing total debt
      try:
        if stock_info_data['data']['country'] == 'China':
          debt.append(int(stock_info_data['data']['totalDebt']/1000000000)*0.16)
        elif stock_info_data['data']['country'] == 'Taiwan':
          debt.append(int(stock_info_data['data']['totalDebt']/1000000000)*0.031)
        else: 
          debt.append(int(stock_info_data['data']['totalDebt']/1000000000))
      except:
        debt.append(0)
      
      #Parsing operating cash
      try:
        if stock_info_data['data']['country'] == 'China':
          op_cash.append(int(stock_info_data['data']['operatingCashflow']/1000000000)*0.16)
        elif stock_info_data['data']['country'] == 'Taiwan':
          op_cash.append(int(stock_info_data['data']['operatingCashflow']/1000000000)*0.031)
        else: 
          op_cash.append(int(stock_info_data['data']['operatingCashflow']/1000000000))
      except: 
        op_cash.append(np.nan)
      
      #Parsing free cash
      try:
        if stock_info_data['data']['country'] == 'China':
          free_cash.append(int(stock_info_data['data']['freeCashflow']/1000000000)*0.16)
        elif stock_info_data['data']['country'] == 'Taiwan':
          free_cash.append(int(stock_info_data['data']['freeCashflow']/1000000000)*0.031)
        else: 
          free_cash.append(int(stock_info_data['data']['freeCashflow']/1000000000))
      except: 
        free_cash.append(np.nan)
      
      #Parsing EBITDA
      try:
        if stock_info_data['data']['country'] == 'China':
          ebitda.append(int(stock_info_data['data']['ebitda']/1000000000)*0.16)
        elif stock_info_data['data']['country'] == 'Taiwan':
          ebitda.append(int(stock_info_data['data']['ebitda']/1000000000)*0.031)
        else: 
          ebitda.append(int(stock_info_data['data']['ebitda']/1000000000))
      except: 
        ebitda.append(np.nan)      
      
      #Parsing dividend yield
      try:
        div_yield.append(np.round(stock_info_data['data']['dividendYield']*100,2))
      except:
        div_yield.append(np.nan)
      
      #Parsing operating margin
      try:
        op_margin.append(np.round(stock_info_data['data']['operatingMargins']*100,2))
      except:
        op_margin.append(np.nan)
      
      #Parsing price to book ratio
      try:
        price_book_ratio.append(np.round(stock_info_data['data']['priceToBook'],2))
      except:
        price_book_ratio.append(np.nan)
      
      #Parsing PEG ratio
      try:
        peg_ratio.append(np.round(stock_info_data['data']['pegRatio'],2))
      except:
        peg_ratio.append(np.nan)
      
      #Parsing debt to equity ratio
      try:
        debt_equity_ratio.append(np.round(stock_info_data['data']['debtToEquity'],2))
      except:
        debt_equity_ratio.append(np.nan)
        
      #Parsing trailing EPS
      try:
        trailing_eps.append(np.round(stock_info_data['data']['trailingEps'],2))
      except:
        trailing_eps.append(np.nan)
        
      #Parsing forward EPS
      try:
        forward_eps.append(np.round(stock_info_data['data']['forwardEps'],2))
      except:
        forward_eps.append(np.nan)
        
      #Parsing return on equity
      try:
        return_on_equity.append(np.round(stock_info_data['data']['returnOnEquity']*100,2))
      except:
        return_on_equity.append(np.nan)
        
      #Parsing return on assets
      try:
        return_on_assets.append(np.round(stock_info_data['data']['returnOnAssets']*100,2))
      except:
        return_on_assets.append(np.nan)
    else: 
      missing_ticker.append(tick)
      bad_response.append(response_status)
      continue
        
    #Parsing stock info metrics
    if earnings_code == 200:
        #Revenue calculation
      yearly_revenue_data = earnings_data['data']
      yearly_revenue_data.reverse()
      
      #quarterly_revenue_data = data['earnings']['financialsChart']['quarterly']
      #quarterly_revenue_data.reverse()

      rev1 = []
      rev2 = []
      quarter = []
      year = []
      
      #Parsing yearly revenue 
      for yearly_rev in yearly_revenue_data:
        #Checking if the revenue needs to be converted from Yuan to USD
        if stock_info_data['data']['country'] == 'China':
          rev1.append(yearly_rev['Revenue']*0.16)
        elif stock_info_data['data']['country'] == 'Taiwan':
          rev1.append(yearly_rev['Revenue']*0.031)
        else: 
          rev1.append(yearly_rev['Revenue'])
        year.append(yearly_rev['Year'])
      try:
        rev_year1.append(rev1[count + 3])
      except:
        rev_year1.append(np.nan)
      try:
        rev_year2.append(rev1[count + 2])
      except: 
        rev_year2.append(np.nan)
      try:
        rev_year3.append(rev1[count + 1])
      except:
        rev_year3.append(np.nan)
      try:
        rev_year4.append(rev1[count])
      except: 
        rev_year4.append(np.nan)
      
      '''
      for quarterly_rev in quarterly_revenue_data:
        #Checking if the revenue needs to be converted from Yuan to USD
        if data['summaryProfile']['country'] == 'China':
          rev2.append(quarterly_rev['revenue']['raw']*0.16)
        elif data['summaryProfile']['country'] == 'Taiwan':
          rev2.append(yearly_rev['revenue']['raw']*0.031)
        else: 
          rev2.append(quarterly_rev['revenue']['raw'])
        quarter.append(quarterly_rev['date'])
      try:
        rev_quarter1.append(rev2[count + 3])
      except:
        rev_quarter1.append(np.nan)
      try:
        rev_quarter2.append(rev2[count + 2])
      except: 
        rev_quarter2.append(np.nan)
      try:
        rev_quarter3.append(rev2[count + 1])
      except:
        rev_quarter3.append(np.nan)
      try:
        rev_quarter4.append(rev2[count])
      except: 
        rev_quarter4.append(np.nan)
      '''
      
      #Profit/earnings calculation
      earnings1 = []
      earnings2 = []
      year = []
      quarter = []
      
      #Parsing yearly earnings
      for yearly_rev in yearly_revenue_data:
        #Checking if the revenue needs to be converted from Yuan to USD
        if stock_info_data['data']['country'] == 'China':
          earnings1.append(yearly_rev['Earnings']*0.16)
        elif stock_info_data['data']['country'] == 'Taiwan':
          earnings1.append(yearly_rev['Earnings']*0.031)
        else: 
          earnings1.append(yearly_rev['Earnings'])
        year.append(yearly_rev['Year'])
      try:
        earnings_year1.append(earnings1[count + 3])
      except:
        earnings_year1.append(np.nan)
      try:
        earnings_year2.append(earnings1[count + 2])
      except: 
        earnings_year2.append(np.nan)
      try:
        earnings_year3.append(earnings1[count + 1])
      except:
        earnings_year3.append(np.nan)
      try:
        earnings_year4.append(earnings1[count])
      except: 
        earnings_year4.append(np.nan)
        
      '''
      for quarterly_rev in quarterly_revenue_data:
        #Checking if the revenue needs to be converted from Yuan to USD
        if data['summaryProfile']['country'] == 'China':
          earnings2.append(quarterly_rev['earnings']['raw']*0.16)
        elif data['summaryProfile']['country'] == 'Taiwan':
          earnings2.append(yearly_rev['revenue']['raw']*0.031)
        else: 
          earnings2.append(quarterly_rev['earnings']['raw'])
        year.append(quarterly_rev['date'])
      try:
        earnings_quarter1.append(earnings2[count + 3])
      except:
        earnings_quarter1.append(np.nan)
      try:
        earnings_quarter2.append(earnings2[count + 2])
      except: 
        earnings_quarter2.append(np.nan)
      try:
        earnings_quarter3.append(earnings2[count + 1])
      except:
        earnings_quarter3.append(np.nan)
      try:
        earnings_quarter4.append(earnings2[count])
      except: 
        earnings_quarter4.append(np.nan)
      '''
    else: 
      missing_ticker.append(tick)
      bad_response.append(response_status)
      continue
    
    '''
    if quarterly_earnings_code == 200: 
      #Parsing previous quarter
      try: 
        previousQuarter = quarterly_earnings_data['data'][-1]['Quarter']
        print ('Previous quarter:','Q' + previousQuarter[0])
        prev_quarter.append('Q' + previousQuarter[0])
      except:
        prev_quarter.append(np.nan)
      
      #Parsing current quarter
      try: 
        currentQuarter = pd.to_datetime(previousQuarter) + timedelta(days = 100)
        print ('Current quarter:','Q' + str(currentQuarter.quarter))
        current_quarter.append('Q' + str(currentQuarter.quarter))
      except:
        current_quarter.append(np.nan)
      
      #Parsing next quarter
      try: 
        nextQuarter = currentQuarter + timedelta(days = 100)
        print ('Next quarter:','Q' + str(nextQuarter.quarter))
        next_quarter.append('Q' + str(nextQuarter.quarter))
      except:
        next_quarter.append(np.nan)
    else: 
      missing_ticker.append(tick)
      bad_response.append(response_status)
      continue
    '''
  
  #Building dataframe with metrics for all tickers
  df['Company'] = company_name
  df['Ticker'] = tickers
  df['Price($)'] = price
  df['Market Cap($)'] = market_cap
  df['Trailing P/E'] = trailing_pe
  df['Forward P/E'] = forward_pe
  df['Rev Yr 1($)'] = rev_year1
  df['Rev Yr 2($)'] = rev_year2
  df['Rev Yr 3($)'] = rev_year3
  df['Rev Yr 4($)'] = rev_year4
  df['Profit Yr 1($)'] = earnings_year1
  df['Profit Yr 2($)'] = earnings_year2
  df['Profit Yr 3($)'] = earnings_year3
  df['Profit Yr 4($)'] = earnings_year4
  '''
  df['Rev Q1($)'] = rev_quarter1
  df['Rev Q2($)'] = rev_quarter2
  df['Rev Q3($)'] = rev_quarter3
  df['Rev Q4($)'] = rev_quarter4
  df['Profit Q1($)'] = earnings_quarter1
  df['Profit Q2($)'] = earnings_quarter2
  df['Profit Q3($)'] = earnings_quarter3
  df['Profit Q4($)'] = earnings_quarter4
  '''
  df['Profit Margin(%)'] = profit_margin
  df['Cash($)'] = cash 
  df['Debt($)'] = debt
  df['Operating Cash($)'] = op_cash
  df['Free Cash($)'] = free_cash
  df['EBITDA'] = ebitda
  df['P/S Ratio'] = ps_ratio
  df['Revenue Growth Year 3(%)'] = ((df['Rev Yr 3($)'] - df['Rev Yr 2($)'])/df['Rev Yr 2($)'])*100
  df['Revenue Growth Year 4(%)'] = ((df['Rev Yr 4($)'] - df['Rev Yr 3($)'])/df['Rev Yr 3($)'])*100
  df['Profit Growth Year 4(%)'] = ((df['Profit Yr 4($)'] - df['Profit Yr 3($)'])/df['Profit Yr 3($)'])*100
  df['Profit Growth Year 3(%)'] = ((df['Profit Yr 3($)'] - df['Profit Yr 2($)'])/df['Profit Yr 2($)'])*100
  #df['Current Quarter'] = current_quarter
  #df['Earnings Date'] = earnings_date
  df['Div Yield(%)'] = div_yield
  df['Operating Margin(%)'] = op_margin
  df['PEG Ratio'] = peg_ratio
  df['Price/Book Ratio'] = price_book_ratio
  df['Debt/Equity Ratio'] = debt_equity_ratio
  df['Trailing EPS'] = trailing_eps
  df['Forward EPS'] = forward_eps
  df['Return On Equity(%)'] = return_on_equity
  df['Return On Assets(%)'] = return_on_assets
      
  #Setting the ticker symbol as the index
  df.index = df['Ticker']
  df = df.drop('Ticker',axis=1)
  
  #Handling tickers that gave a bad API response
  if len(missing_ticker) > 0: 
    print ("The following tickers returned a bad response:")
    for i,j in zip(missing_ticker,bad_response): 
      print (i,j)
  else: 
    pass
  
  return df

def plot_dash(watchlist):

  plt.rcParams.update(plt.rcParamsDefault)
  #   %matplotlib inline
  plt.style.use('bmh')

  fig3 = plt.figure(constrained_layout=True,figsize=(42,30))
  gs = fig3.add_gridspec(60,3)
  
  f3_ax2 = fig3.add_subplot(gs[0:19,0])
  plt.bar(np.array(range(len(watchlist.index)))-0.2,watchlist['Rev Yr 2($)'],width=0.2,label="Revenue year 2",alpha=0.8)
  plt.bar(np.array(range(len(watchlist.index))),watchlist['Rev Yr 3($)'],width=0.2,label="Revenue year 3",alpha=0.8)
  plt.bar(np.array(range(len(watchlist.index)))+0.2,watchlist['Rev Yr 4($)'],width=0.2,label="Revenue year 4",alpha=0.8)
  plt.xticks(range(len(watchlist.index)),list(watchlist.index),size=25,rotation=45)
  max_revenue = np.max([watchlist['Rev Yr 2($)'].max(),watchlist['Rev Yr 3($)'].max(),watchlist['Rev Yr 4($)'].max()])
  if (max_revenue > 50000000000) & (max_revenue < 100000000000):
    plt.yticks(range(0,120000000000,20000000000),size=25)
  elif (max_revenue < 10000000000):
    plt.yticks(range(0,20000000000,10000000000),size=25)
  elif (max_revenue < 20000000000) & (max_revenue > 10000000000) :
    plt.yticks(range(0,30000000000,10000000000),size=25)
  elif (max_revenue < 30000000000) & (max_revenue > 20000000000) :
    plt.yticks(range(0,40000000000,10000000000),size=25)
  elif (max_revenue < 40000000000) & (max_revenue > 30000000000) :
    plt.yticks(range(0,50000000000,10000000000),size=25)
  elif (max_revenue < 50000000000) & (max_revenue > 40000000000) :
    plt.yticks(range(0,60000000000,10000000000),size=25)
  elif (max_revenue < 60000000000) & (max_revenue > 50000000000) :
    plt.yticks(range(0,70000000000,10000000000),size=25)
  elif (max_revenue < 70000000000) & (max_revenue > 60000000000) :
    plt.yticks(range(0,80000000000,10000000000),size=25)
  elif (max_revenue < 80000000000) & (max_revenue > 70000000000) :
    plt.yticks(range(0,90000000000,10000000000),size=25)
  else: 
    plt.yticks(range(0,60000000000,10000000000),size=25)
  plt.xlabel('Ticker',size=35)
  plt.ylabel('Revenue (10 Billion $)',size=35)
  
  plt.title('Revenue Last 3 Years',size=40)
  plt.legend(prop={'size':30})

  f3_ax4 = fig3.add_subplot(gs[0:19,1])  
  plt.bar(np.array(range(len(watchlist.index)))-0.2,watchlist['Revenue Growth Year 3(%)'],width=0.4,label="Rev growth previous year",alpha=0.8)
  plt.bar(np.array(range(len(watchlist.index)))+0.2,watchlist['Revenue Growth Year 4(%)'],width=0.4,label="Rev growth current year",alpha=0.8)
  plt.xlabel('Ticker',size=35)
  plt.ylabel('Revenue growth (%)',size=35)
  plt.xticks(range(len(watchlist.index)),list(watchlist.index),size=25,rotation=45)
  plt.yticks(size=25)
  plt.title('Revenue Growth',size=40)
  plt.legend(prop={'size':30})
  
  '''
  f3_ax3 = fig3.add_subplot(gs[0:19,2])
  plt.bar(np.array(range(len(watchlist.index)))-0.2,watchlist['Rev Q4($)'],width=0.4,label="Revenue current quarter",alpha=0.8)
  plt.bar(np.array(range(len(watchlist.index)))+0.2,watchlist['Profit Q4($)'],width=0.4,label="Earnings current quarter",alpha=0.8)
  plt.xlabel('Ticker',size=35)
  plt.ylabel('Dollar (10 Billion $)',size=35)
  plt.xticks(range(len(watchlist.index)),list(watchlist.index),size=25,rotation=45)
  max_val = np.max([watchlist['Rev Q4($)'].max(),watchlist['Profit Q4($)'].max()])
  if max_val > 50000000000:
    plt.yticks(range(0,120000000000,20000000000),size=25)
  else: 
    plt.yticks(range(0,60000000000,10000000000),size=25)
  plt.title('Revenue And Earnings Current Quarter',size=40)
  plt.legend(prop={'size':30})
  '''
  
  f3_ax3 = fig3.add_subplot(gs[0:19,2])
  plt.bar(np.array(range(len(watchlist.index)))-0.2,watchlist['Trailing P/E'],width=0.4,label="Trailing P/E",alpha=0.8,color='orange')
  plt.bar(np.array(range(len(watchlist.index)))+0.2,watchlist['Forward P/E'],width=0.4,label="Forward P/E",alpha=0.8,color='green')
  plt.xlabel('Ticker',size=35)
  plt.ylabel('P/E Ratio',size=35)
  plt.xticks(range(len(watchlist.index)),list(watchlist.index),size=25,rotation=45)
  plt.yticks(size=25)
  plt.ylim(0,100)
  plt.title('Trailing and Forward P/E Ratios',size=40)
  plt.legend(prop={'size':30})
  
  f3_ax3 = fig3.add_subplot(gs[21:40,2])
  plt.scatter(watchlist.index,watchlist['Profit Margin(%)'],s=4000,color='green',label='Profit Margin',alpha=0.8)
  for ind,point in enumerate(watchlist['Profit Margin(%)']):
    plt.annotate(str(point) + '%',xy=(ind,point+1.5),xytext=(ind,point+1.5),size=30)
    
  plt.scatter(watchlist.index,watchlist['Operating Margin(%)'],s=4000,color='orange',label='Operating Margin',alpha=0.8)
  for ind,point in enumerate(watchlist['Operating Margin(%)']):
    plt.annotate(str(point) + '%',xy=(ind,point+1.5),xytext=(ind,point+1.5),size=30)
    
  plt.yticks(size=25)
  plt.xticks(rotation=45,size=25)
  plt.ylabel('Margin (%)',size=35)
  plt.xlabel('Ticker',size=35)
  plt.title('Margins',size=40)
  min_val = np.min([watchlist['Profit Margin(%)'].min(),watchlist['Operating Margin(%)'].min()])
  max_val = np.max([watchlist['Profit Margin(%)'].max(),watchlist['Operating Margin(%)'].max()])
  plt.ylim(0, max_val + 5)
  plt.legend(prop={'size':30})

  f3_ax4 = fig3.add_subplot(gs[42:60,1])  
  plt.bar(np.array(range(len(watchlist['Cash($)'])))-0.2,watchlist['Cash($)'],width=0.4,label="Cash on hand",alpha=0.8)
  plt.bar(np.array(range(len(watchlist['Debt($)'])))+0.2,watchlist['Debt($)'],width=0.4,label="Total debt",alpha=0.8)
  plt.xlabel("Ticker",size=35)
  plt.ylabel("Dollar (Billion $)",size=35)
  plt.xticks(range(len(watchlist.index)),list(watchlist.index),size=25,rotation=45)
  plt.yticks(size=25)
  plt.title('Cash On Hand And Total Debt',size=40)
  plt.legend(prop={'size':30})

  f3_ax4 = fig3.add_subplot(gs[42:60,2])  
  plt.bar(np.array(range(len(watchlist['Operating Cash($)'])))-0.2,watchlist['Operating Cash($)'],width=0.4,label="Operating cash",color='orange',alpha=0.8)
  plt.bar(np.array(range(len(watchlist['Free Cash($)'])))+0.2,watchlist['Free Cash($)'],width=0.4,label="Free cash",color='green',alpha=0.8)
  plt.xlabel("Ticker",size=35)
  plt.ylabel("Cash (Billion $)",size=35)
  plt.xticks(range(len(watchlist.index)),list(watchlist.index),size=25,rotation=45)
  plt.yticks(size=25)
  plt.title('Operating Cash And Free Cash',size=40)
  plt.legend(prop={'size':30})

  f3_ax4 = fig3.add_subplot(gs[42:60,0])  
  plt.bar(np.array(range(len(watchlist['EBITDA']))),watchlist['EBITDA'],width=0.5,label="EBITDA",color='purple',alpha=0.8)
  plt.xlabel("Ticker",size=35)
  plt.ylabel("EBITDA",size=35)
  plt.xticks(range(len(watchlist.index)),list(watchlist.index),size=25,rotation=45)
  plt.yticks(size=25)
  plt.title('EBITDA',size=40)
  #plt.legend(prop={'size':30})
  
  f3_ax2 = fig3.add_subplot(gs[21:40,0])
  plt.bar(np.array(range(len(watchlist.index)))-0.2,watchlist['Profit Yr 2($)'],width=0.2,label="Earnings year 2",alpha=0.8)
  plt.bar(np.array(range(len(watchlist.index))),watchlist['Profit Yr 3($)'],width=0.2,label="Earnings year 3",alpha=0.8)
  plt.bar(np.array(range(len(watchlist.index)))+0.2,watchlist['Profit Yr 4($)'],width=0.2,label="Earnings year 4",alpha=0.8)
  plt.xticks(range(len(watchlist.index)),list(watchlist.index),size=25,rotation=45)
  max_profit = np.max([watchlist['Profit Yr 2($)'].max(),watchlist['Profit Yr 3($)'].max(),watchlist['Profit Yr 4($)'].max()])
  if (max_profit > 50000000000) & (max_profit < 100000000000):
    plt.yticks(range(0,120000000000,20000000000),size=25)
  elif (max_profit < 10000000000):
    plt.yticks(range(0,20000000000,10000000000),size=25)
  elif (max_profit < 20000000000) & (max_profit > 10000000000) :
    plt.yticks(range(0,30000000000,10000000000),size=25)
  elif (max_profit < 30000000000) & (max_profit > 20000000000) :
    plt.yticks(range(0,40000000000,10000000000),size=25)
  elif (max_profit < 40000000000) & (max_profit > 30000000000) :
    plt.yticks(range(0,50000000000,10000000000),size=25)
  elif (max_profit < 50000000000) & (max_profit > 40000000000) :
    plt.yticks(range(0,60000000000,10000000000),size=25)
  elif (max_profit < 60000000000) & (max_profit > 50000000000) :
    plt.yticks(range(0,70000000000,10000000000),size=25)
  elif (max_profit < 70000000000) & (max_profit > 60000000000) :
    plt.yticks(range(0,80000000000,10000000000),size=25)
  elif (max_profit < 80000000000) & (max_profit > 70000000000) :
    plt.yticks(range(0,90000000000,10000000000),size=25)
  else: 
    plt.yticks(range(0,60000000000,10000000000),size=25)
  plt.xlabel('Ticker',size=35)
  plt.ylabel('Profit (10 Billion $)',size=35)
  plt.title('Earnings Last 3 Years',size=40)
  plt.legend(prop={'size':30})
  
  f3_ax4 = fig3.add_subplot(gs[21:40,1])  
  plt.bar(np.array(range(len(watchlist.index)))-0.2,watchlist['Profit Growth Year 3(%)'],width=0.4,label="Profit growth previous year",alpha=0.8)
  plt.bar(np.array(range(len(watchlist.index)))+0.2,watchlist['Profit Growth Year 4(%)'],width=0.4,label="Profit growth current year",alpha=0.8)
  plt.xlabel('Ticker',size=35)
  plt.ylabel('Profit growth (%)',size=35)
  plt.xticks(range(len(watchlist.index)),list(watchlist.index),size=25,rotation=45)
  plt.yticks(size=25)
  plt.title('Profit Growth',size=40)
  plt.legend(prop={'size':30})

  #plt.tight_layout(pad=4)
  st.pyplot(fig3)

  #plt.show()