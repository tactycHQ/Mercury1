#MERCURY 1
import logging
import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s-%(process)d-%(levelname)s-%(message)s',datefmt='%d-%b-%y %H:%M:%S',stream=sys.stdout)

#GLOBAL VARIABLES
__AAPL__ = "data\ciq_aapl_values.csv"

selectedFeatures = ['DATE',
'IQ_LASTSALEPRICE',
'IQ_PRICEDATE',
'IQ_HIGHPRICE',
'IQ_LOWPRICE',
'IQ_OPENPRICE',
'IQ_CLOSEPRICE',
'IQ_CLOSEPRICE_ADJ',
'IQ_VWAP',
'IQ_EQUITY_BIDPRICE',
'IQ_EQUITY_ASKPRICE',
'IQ_EQUITY_MIDPRICE',
'IQ_YEARHIGH',
'IQ_YEARLOW',
'IQ_VOLUME',
'IQ_VALUE_TRADED',
'IQ_MARKETCAP',
'IQ_CLASS_MARKETCAP',
'IQ_TEV',
'IQ_SHARESOUTSTANDING',
'IQ_CLASS_SHARESOUTSTANDING',
'IQ_BETA_5YR',
'IQ_BETA_2YR',
'IQ_BETA_1YR',
'IQ_BETA_5YR_RSQ',
'IQ_BETA_2YR_RSQ',
'IQ_BETA_1YR_RSQ',
'IQ_PRICE_VOL_HIST_5YR',
'IQ_PRICE_VOL_HIST_2YR',
'IQ_PRICE_VOL_HIST_YR',
'IQ_PRICE_VOL_HIST_6MTH',
'IQ_PRICE_VOL_HIST_3MTH',
'IQ_DIVIDEND_YIELD',
'IQ_ANNUALIZED_DIVIDEND',
'IQ_SHORT_INTEREST',
'IQ_SHORT_INTEREST_PERCENT',
'IQ_DIV_AMOUNT',
'IQ_NEXT_DIV_AMOUNT',
'IQ_VOL_LAST_WK',
'IQ_VOL_LAST_MTH',
'IQ_VOL_LAST_3MTH',
'IQ_VOL_LAST_6MTH',
'IQ_VOL_LAST_YR',
'IQ_VALUE_TRADED_LAST_WK',
'IQ_VALUE_TRADED_LAST_MTH',
'IQ_VALUE_TRADED_LAST_3MTH',
'IQ_VALUE_TRADED_LAST_6MTH',
'IQ_VALUE_TRADED_LAST_YR',
'IQ_RSI',
'IQ_RSI_ADJ',
'IQ_RETURN_ASSETS',
'IQ_RETURN_CAPITAL',
'IQ_RETURN_INVESTED_CAPITAL',
'IQ_RETURN_EQUITY',
'IQ_RETURN_COMMON_EQUITY',
'IQ_GROSS_MARGIN',
'IQ_SGA_MARGIN',
'IQ_EBITDA_MARGIN',
'IQ_EBITA_MARGIN',
'IQ_EBIT_MARGIN',
'IQ_NI_NORM_MARGIN',
'IQ_ASSET_TURNS',
'IQ_CURRENT_RATIO',
'IQ_QUICK_RATIO',
'IQ_Z_SCORE',
'IQ_TOTAL_REV_1YR_ANN_GROWTH',
'IQ_GP_1YR_ANN_GROWTH',
'IQ_EBITA_1YR_ANN_GROWTH',
'IQ_EBIT_1YR_ANN_GROWTH',
'IQ_NI_NORM_1YR_ANN_GROWTH',
'IQ_DPS_1YR_ANN_GROWTH',
'IQ_ACCT_RECV_1YR_ANN_GROWTH',
'IQ_COMMON_EQUITY_1YR_ANN_GROWTH',
'IQ_TBV_1YR_ANN_GROWTH',
'IQ_TOTAL_ASSETS_1YR_ANN_GROWTH',
'IQ_TOTAL_REV_2YR_ANN_CAGR',
'IQ_GP_2YR_ANN_CAGR',
'IQ_EBITA_2YR_ANN_CAGR',
'IQ_EBIT_2YR_ANN_CAGR',
'IQ_NI_NORM_2YR_ANN_CAGR',
'IQ_DPS_2YR_ANN_CAGR',
'IQ_ACCT_RECV_2YR_ANN_CAGR',
'IQ_COMMON_EQUITY_2YR_ANN_CAGR',
'IQ_TBV_2YR_ANN_CAGR',
'IQ_TOTAL_ASSETS_2YR_ANN_CAGR',
'IQ_TOTAL_REV_3YR_ANN_CAGR',
'IQ_GP_3YR_ANN_CAGR',
'IQ_EBITA_3YR_ANN_CAGR',
'IQ_EBIT_3YR_ANN_CAGR',
'IQ_NI_NORM_3YR_ANN_CAGR',
'IQ_DPS_3YR_ANN_CAGR',
'IQ_ACCT_RECV_3YR_ANN_CAGR',
'IQ_COMMON_EQUITY_3YR_ANN_CAGR',
'IQ_TBV_3YR_ANN_CAGR',
'IQ_TOTAL_ASSETS_3YR_ANN_CAGR',
'IQ_TOTAL_REV_5YR_ANN_CAGR',
'IQ_GP_5YR_ANN_CAGR',
'IQ_NI_NORM_5YR_ANN_CAGR',
'IQ_DPS_5YR_ANN_CAGR',
'IQ_ACCT_RECV_5YR_ANN_CAGR',
'IQ_COMMON_EQUITY_5YR_ANN_CAGR',
'IQ_TBV_5YR_ANN_CAGR',
'IQ_TOTAL_ASSETS_5YR_ANN_CAGR',
'IQ_TOTAL_REV_7YR_ANN_CAGR',
'IQ_GP_7YR_ANN_CAGR',
'IQ_ACCT_RECV_7YR_ANN_CAGR',
'IQ_COMMON_EQUITY_7YR_ANN_CAGR',
'IQ_TBV_7YR_ANN_CAGR',
'IQ_TOTAL_ASSETS_7YR_ANN_CAGR',
'IQ_TOTAL_REV_10YR_ANN_CAGR',
'IQ_GP_10YR_ANN_CAGR',
'IQ_DPS_10YR_ANN_CAGR',
'IQ_ACCT_RECV_10YR_ANN_CAGR',
'IQ_COMMON_EQUITY_10YR_ANN_CAGR',
'IQ_TBV_10YR_ANN_CAGR',
'IQ_TOTAL_ASSETS_10YR_ANN_CAGR',
'IQ_TEV_TOTAL_REV',
'IQ_TEV_EBIT',
'IQ_TEV_EBIT_OUT',
'IQ_TEV_EBIT_EXER',
'IQ_PRICE_SALES',
'IQ_PE_NORMALIZED',
'IQ_PBV',
'IQ_PTBV',
'IQ_MKTCAP_TOTAL_REV',
'IQ_MKTCAP_TOTAL_REV_OUT',
'IQ_MKTCAP_TOTAL_REV_EXER',
'IQ_MKTCAP_EBT_EXCL',
'IQ_MKTCAP_EBT_EXCL_OUT',
'IQ_MKTCAP_EBT_EXCL_EXER',
'IQ_EST_NUM_HIGH_REC',
'IQ_EST_NUM_NEUTRAL_REC',
'IQ_PRICE_TARGET',
'IQ_MEDIAN_TARGET_PRICE',
'IQ_HIGH_TARGET_PRICE',
'IQ_LOW_TARGET_PRICE',
'IQ_TARGET_PRICE_NUM']

class DataGenerator:

    def __init__(self,fname, window=1,threshold=0.05):

        df = self.createDf(fname)
        self.window=1
        self.threshold=threshold
        self.dates=None
        self.prices = df['IQ_LASTSALEPRICE'].values.reshape(-1, 1)
        self.createInputs(df)
        self.createTargets()
        self.splitData()
        self.getNormalizeData()

    def createDf(self,fname):
        df = pd.read_csv(fname,low_memory=False)
        logging.info("Creating Dataframe from CSV")

        df['DATE'] = pd.to_datetime(df['DATE'])
        self.dates = df.loc[:, 'DATE'].values.reshape(-1, 1)
        logging.info("Date Reformatted")

        df = df[selectedFeatures]
        df.describe(include='all').to_csv("data\\Unscaled Feature Description.csv")
        logging.info("Unscaled Features Description Saved Under Feature Description.csv")
        return df

    def createInputs(self,df):
        self.inputs = df.loc[:, df.columns != 'DATE'].values
        logging.info("Inputs created of shape %s",self.inputs.shape)


    def createTargets(self):
        pctReturns = self.createPctReturns()
        targets = []
        for i in range (0,len(pctReturns)):
            if pctReturns[i]>self.threshold:
                targets.append(1)
            elif pctReturns[i] < -self.threshold:
                    targets.append(-1)
            else:
                targets.append(0)
        self.targets = np.array(targets)

        logging.info("Targets created of shape %s", self.targets.shape)
        unique, counts = np.unique(self.targets,return_counts=True)
        logging.info("Target counts are %s %s",unique,counts)


    def createPctReturns(self):
        len = self.prices.shape[0]
        pctReturns = np.empty((len, 1))
        for i in range (0,len-self.window):
            pctReturns[i] = self.prices[i+self.window,0]/self.prices[i,0]-1
        return pctReturns

    def splitData(self):
        self.X_train, self.X_test, Y_train,Y_test = train_test_split(self.inputs,self.targets,test_size=0.2,random_state=1,stratify=None)
        self.Y_train=np.reshape(Y_train,(-1, 1))
        self.Y_test=np.reshape(Y_test,(-1, 1))
        logging.info("Train and test sets have been split")


    def getNormalizeData(self):
        sc = StandardScaler()
        sc.fit(self.X_train)
        self.X_train_std = sc.transform(self.X_train)
        self.X_test_std = sc.transform(self.X_test)

        logging.info("Train and test sets have been normalized")
        logging.info("X_train_std shape is %s", self.X_train_std.shape)
        logging.info("X_test_std is %s", self.X_test_std.shape)
        logging.info("Y_train shape is %s", self.Y_train.shape)
        logging.info("Y_test shape is %s", self.Y_test.shape)

df_aapl = DataGenerator(__AAPL__,1,0.05)











