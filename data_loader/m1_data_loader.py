#MERCURY 1
import logging
import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s-%(process)d-%(levelname)s-%(message)s',datefmt='%d-%b-%y %H:%M:%S',stream=sys.stdout)

class DataLoader:

    def __init__(self,fname, window=1,threshold=0.05):

        self.drop_features = [
            'DATE',
            'IQ_LASTSALEPRICE',
            'BENCHMARK',
            'IQ_LOWPRICE',
            'IQ_OPENPRICE',
            'IQ_CLOSEPRICE',
            'IQ_CLOSEPRICE_ADJ',
            'IQ_VWAP',
            'IQ_EQUITY_BIDPRICE',
            'IQ_EQUITY_ASKPRICE',
            'IQ_EQUITY_MIDPRICE',
            'IQ_YEARLOW',
            'IQ_MARKETCAP',
            'IQ_CLASS_MARKETCAP',
            'IQ_TEV',
            'IQ_CLASS_SHARESOUTSTANDING',
            'IQ_SHORT_INTEREST_PERCENT',
            'IQ_VOL_LAST_MTH',
            'IQ_VOL_LAST_3MTH',
            'IQ_VOL_LAST_6MTH',
            'IQ_VOL_LAST_YR',
            'IQ_VALUE_TRADED_LAST_WK',
            'IQ_VALUE_TRADED_LAST_MTH',
            'IQ_VALUE_TRADED_LAST_3MTH',
            'IQ_VALUE_TRADED_LAST_6MTH',
            'IQ_VALUE_TRADED_LAST_YR',
            'IQ_RSI_ADJ',
            'IQ_RETURN_COMMON_EQUITY',
            'IQ_EBIT_MARGIN',
            'IQ_NI_NORM_MARGIN',
            'IQ_QUICK_RATIO',
            'IQ_EBIT_1YR_ANN_GROWTH',
            'IQ_NI_NORM_1YR_ANN_GROWTH',
            'IQ_TBV_1YR_ANN_GROWTH',
            'IQ_EBIT_2YR_ANN_CAGR',
            'IQ_NI_NORM_2YR_ANN_CAGR',
            'IQ_TBV_2YR_ANN_CAGR',
            'IQ_EBIT_3YR_ANN_CAGR',
            'IQ_NI_NORM_3YR_ANN_CAGR',
            'IQ_TBV_3YR_ANN_CAGR',
            'IQ_TOTAL_ASSETS_3YR_ANN_CAGR',
            'IQ_GP_5YR_ANN_CAGR',
            'IQ_COMMON_EQUITY_5YR_ANN_CAGR',
            'IQ_TBV_5YR_ANN_CAGR',
            'IQ_TOTAL_ASSETS_5YR_ANN_CAGR',
            'IQ_GP_7YR_ANN_CAGR',
            'IQ_TBV_7YR_ANN_CAGR',
            'IQ_GP_10YR_ANN_CAGR',
            'IQ_TBV_10YR_ANN_CAGR',
            'IQ_TEV_EBIT_EXER',
            'IQ_PE_NORMALIZED',
            'IQ_PTBV',
            'IQ_MKTCAP_TOTAL_REV',
            'IQ_MKTCAP_TOTAL_REV_EXER',
            'IQ_MKTCAP_EBT_EXCL',
            'IQ_MKTCAP_EBT_EXCL_OUT',
            'IQ_MKTCAP_EBT_EXCL_EXER',
            'IQ_PRICE_TARGET',
            'IQ_MEDIAN_TARGET_PRICE',
            'IQ_HIGH_TARGET_PRICE',
            'IQ_VOL_LAST_WK',
            'IQ_DPS_10YR_ANN_CAGR',
            'IQ_EBITA_2YR_ANN_CAGR',
            'IQ_EBITA_MARGIN',
            'IQ_COMMON_EQUITY_3YR_ANN_CAGR',
            'IQ_EBITDA_MARGIN',
            'IQ_DPS_1YR_ANN_GROWTH',
            'IQ_TOTAL_ASSETS_7YR_ANN_CAGR',
            'IQ_NI_NORM_5YR_ANN_CAGR',
            'IQ_GP_3YR_ANN_CAGR',
            'IQ_TOTAL_REV_7YR_ANN_CAGR',
            'IQ_COMMON_EQUITY_7YR_ANN_CAGR',
            'IQ_DPS_3YR_ANN_CAGR',
            'IQ_DIV_AMOUNT',
            'IQ_DPS_5YR_ANN_CAGR',
            'IQ_ANNUALIZED_DIVIDEND'
        ]
        self.dates = None
        self.features=None
        self.targets=None
        self.targets_train=None
        df = self.createDf(fname)
        self.window=window
        self.threshold=threshold
        self.prices = df['IQ_LASTSALEPRICE'].values.reshape(-1, 1)
        self.bmark = df['BENCHMARK'].values.reshape(-1, 1)
        self.relReturns=None
        self.createInputs(df)
        self.createTargets()
        self.splitData()
        self.getNormalizeData()

    def createDf(self,fname):
        df = pd.read_csv(fname,low_memory=False)
        logging.info("Creating Dataframe from CSV")

        df['DATE'] = pd.to_datetime(df['DATE'])
        self.dates = df.loc[:,'DATE'].values.reshape(-1, 1)
        logging.info("Date Reformatted")

        df.describe(include='all').to_csv("C:\\Users\\anubhav\\Desktop\\Projects\\Mercury1\\outputs\\Unscaled Feature Description")
        logging.info("Unscaled Features Description Saved Under Feature Description.csv")
        return df

    def createInputs(self,dframe):
        dframe = dframe.drop(self.drop_features,axis=1)
        self.df=dframe
        self.features=self.df.columns
        self.inputs = self.df.values
        self.inputs = self.inputs[:-self.window]
        logging.info("Inputs created of shape %s",self.inputs.shape)


    def createTargets(self):
        pctReturns = self.createPctReturns(self.prices)
        bMarkReturns = self.createPctReturns(self.bmark)
        self.relReturns = pctReturns - bMarkReturns

        targets = []
        for ret in self.relReturns:
            if ret>self.threshold:
                targets.append(1)
            elif ret < -self.threshold:
                targets.append(-1)
            else:
                targets.append(0)
        self.targets = np.array(targets).reshape(-1,1)
        unique, counts = np.unique(self.targets, return_counts=True)
        logging.info("Target counts are %s %s", unique, counts)

        ohe = OneHotEncoder(categories='auto')
        self.targets_train = ohe.fit_transform(self.targets).toarray()
        self.targets_train = self.targets_train[:-self.window]
        logging.info("Targets are one hot encoded and transformed to shape %s", self.targets_train.shape)


    def createPctReturns(self,close):
        len = close.shape[0]
        pctReturns = np.empty((len, 1))
        for i in range (0,len-self.window):
            pctReturns[i] = close[i+self.window,0]/close[i,0]-1
        return pctReturns

    def splitData(self):
        self.X_train, self.X_test, Y_train,Y_test = train_test_split(self.inputs,self.targets_train,test_size=0.2,random_state=1,stratify=None)
        self.Y_train=np.reshape(Y_train,(-1, Y_train.shape[1]))
        self.Y_test=np.reshape(Y_test,(-1, Y_test.shape[1]))
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











