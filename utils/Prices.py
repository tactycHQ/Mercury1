
import pandas as pd
import numpy as np
from ta import *
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#GLOBAL VARIABLES
__DATA__ = "Data/AAPL.csv"


def main():
    df = createDf()
    createPlots(df)

def createDf():
    print("----Creating Dataframe from CSV----")
    df = pd.read_csv(__DATA__)
    df = df.iloc[::-1].reset_index(drop=True)
    df = add_all_ta_features(df, "open", "high", "low", "close", "volume", fillna=True)
    print("----TA Featured Added----")
    df['date']=pd.to_datetime(df['date'])
    print("----Date Reformatted----")
    df=addLabels(df,30,0.05)
    # print(df.columns)
    df.to_csv("AAPL Mod.csv")
    return df

def addLabels(df,delta,hurdle):
    print("-----Computing Labels----")
    df['labels']=0
    for i in range(0,len(df['close'])-delta):
        if df.loc[i+delta,'close']/df.loc[i,'close']>=1+hurdle:
            print("End: ")
            print(df.loc[i + delta, 'close'])
            print(df.loc[i+delta, 'date'])
            print("Start:")
            print(+df.loc[i, 'close'])
            print(df.loc[i,'date'])
            df.loc[i,'labels']=1
        elif df.loc[i+delta,'close']/df.loc[i,'close']<=1-hurdle:
            df.loc[i,'labels']=-1
        else:
            df.loc[i, 'labels'] = 0
    print("----Labels Computed and Added to Dataframe----")
    return df


def createPlots(df):
    print("----Plotting----")
    dates=df['date']
    prices=df['close']
    labels = df['labels']
    myFmt = mdates.DateFormatter("%m/%d/%Y")
    fig, axs = plt.subplots(2,1)

    axs[0].plot(dates, prices)
    axs[0].xaxis.set_major_formatter(myFmt)
    axs[0].set_title('Stock Price')
    axs[0].legend()

    axs[1].plot(dates, labels, marker='.',linestyle='None')
    axs[1].xaxis.set_major_formatter(myFmt)
    axs[1].set_title('Classifications')
    axs[1].legend()

    #Show Plot
    fig.autofmt_xdate()
    plt.show()

    return df


if __name__ == '__main__':
    main()
