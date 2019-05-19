#MERCURY 1
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#GLOBAL VARIABLES
__AAPL__ = "D:\Dropbox\9. Data\Mercury Data\ciq_aapl_values.csv"
__SPY__ = "D:\Dropbox\9. Data\Mercury Data\ciq_spy_values.csv"


def main():
    df = createDf(__AAPL__)
    df_benchmark = createDf(__SPY__)
    print(df.shape)
    print(df.iloc[0],[0])

def createDf(ticker):
    print("----Creating Dataframe from CSV----")
    df = pd.read_csv(ticker,low_memory=False)

    print("----Date Reformatted----")
    df['DATE'] = pd.to_datetime(df['DATE'])

    return df

if __name__ == '__main__':
    main()
