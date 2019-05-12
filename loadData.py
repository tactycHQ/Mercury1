
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

#GLOBAL VARIABLES
__DATA__ = "D:\Dropbox\9. Data\Mercury Data\ciq_aapl_values.csv"


def main():
    df = createDf()

def createDf():
    print("----Creating Dataframe from CSV----")
    df = pd.read_csv(__DATA__,low_memory=False)

    print("----Date Reformatted----")
    df['DATE'] = pd.to_datetime(df['DATE'])
    print(df.columns)
    # df = df.iloc[::-1].reset_index(drop=True)

    # df=addLabels(df,30,0.05)
    # # print(df.columns)

    return df

if __name__ == '__main__':
    main()
