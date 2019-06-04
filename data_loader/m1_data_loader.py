#MERCURY 1
import logging
import pandas as pd
import sys
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
logging.basicConfig(level=logging.DEBUG,format='%(asctime)s-%(process)d-%(levelname)s-%(message)s',datefmt='%d-%b-%y %H:%M:%S',stream=sys.stdout)

class DataLoader:

    def __init__(self,fname, window=1,threshold=0.05):

        df = self.createDf(fname)
        self.window=window
        self.threshold=threshold
        self.dates=None
        self.prices = df['IQ_LASTSALEPRICE'].values.reshape(-1, 1)
        self.bmark = df['BENCHMARK'].values.reshape(-1, 1)
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

        df.describe(include='all').to_csv("outputs\\Unscaled Feature Description.csv")
        logging.info("Unscaled Features Description Saved Under Feature Description.csv")
        return df

    def createInputs(self,df):
        self.inputs = df.loc[:, df.columns != 'DATE'].values
        self.inputs = self.inputs[:-self.window]
        logging.info("Inputs created of shape %s",self.inputs.shape)


    def createTargets(self):
        pctReturns = self.createPctReturns(self.prices)
        bMarkReturns = self.createPctReturns(self.bmark)
        relReturns = pctReturns - bMarkReturns

        targets = []
        for i in range (0,len(relReturns)-self.window):
            if relReturns[i]>self.threshold:
                targets.append(1)
            elif relReturns[i] < -self.threshold:
                    targets.append(-1)
            else:
                targets.append(0)
        self.targets = np.array(targets)

        logging.info("Targets created of shape %s", self.targets.shape)
        unique, counts = np.unique(self.targets,return_counts=True)
        logging.info("Target counts are %s %s",unique,counts)


    def createPctReturns(self,close):
        len = close.shape[0]
        pctReturns = np.empty((len, 1))
        for i in range (0,len-self.window):
            pctReturns[i] = close[i+self.window,0]/close[i,0]-1
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











