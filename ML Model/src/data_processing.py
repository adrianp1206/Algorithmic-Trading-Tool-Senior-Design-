import yfinance as yf
import pandas as pd 
import os

def fetch_tsla_data(start_date='2015-01-01', end_date='2024-01-01'):
    """
    Fetch historical stock data for TSLA from Yahoo Finance.
    
    Args:
    start_date: str, The start date for the data fetch.
    end_date: str, The end date for the data fetch.
    
    Returns:
    tsla_data: DataFrame, The historical stock data for TSLA.
    """
    tsla_date = yf.download('TSLA', start=start_date, end=end_date)
    return tsla_date

def calculate_technical_indicators(df):
    """
    Calculate common technical indicators using TA-Lib.
    
    Args:
    df: DataFrame, The stock data.
    
    Returns:
    df: DataFrame, The stock data with added indicators.
    """
    return df

def save_data_to_csv(df, filename='tsla_data.csv'):
    """
    Save the stock data to a CSV file.
    
    Args:
    df: DataFrame, The data to save.
    filename: str, The name of the file to save the data to.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename)
    print(f"Data saved to {filename}")

def load_data_from_csv(filename='tsla_data.csv'):
    """
    Load stock data from a CSV file.
    
    Args:
    filename: str, The name of the CSV file.
    
    Returns:
    df: DataFrame, The loaded data.
    """
    return pd.read_csv(filename)

import numpy as np 
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

def create_lstm_input(df, target_column='Adj Close', lookback=60):
    X, y = [], []

    for i in range(lookback, len(df)):
        X.append(df[target_column].iloc[i-lookback:i].values)
        y.append(df[target_column].iloc[i])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    return X, y

def preprocess_data(df):
    """
    Preprocess the stock data by handling missing values and scaling the data.
    
    Args:
    df: DataFrame, The raw stock data.
    
    Returns:
    df: DataFrame, The preprocessed data.
    """
    df.fillna(method='ffill', inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    
    columns_to_scale = ['Adj Close', 'Open', 'High', 'Low', 'Volume']
    df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

    return df

