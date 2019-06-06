import os
import logging
import numpy as np
from data_loader.m1_data_loader import DataLoader
from models.m1_model import M1Model
from trainer.trainer import Trainer


#GLOBAL VARIABLES
_AAPL = "D:\\Dropbox\\9. Data\\Mercury Data\\CSV\\CIQ_AAPL.csv"
_CMCSA = "D:\\Dropbox\\9. Data\\Mercury Data\\CSV\\CIQ_CMCSA.csv"
_AMD = "D:\\Dropbox\\9. Data\\Mercury Data\\CSV\\CIQ_AMD.csv"


def main(load=0):

    X_train = np.concatenate((data_aapl.X_train_std,
                              data_cmcs.X_train_std,
                              data_amd.X_train_std),
                             axis=0)
    Y_train = np.concatenate((data_aapl.Y_train,
                              data_cmcs.Y_train,
                              data_amd.Y_train),
                             axis=0)
    num_features = X_train.shape[1]

    X_test = np.concatenate((data_aapl.X_test_std,
                             data_cmcs.X_test_std,
                            data_amd.X_test_std),
                            axis=0)
    Y_test = np.concatenate((data_aapl.Y_test,
                             data_cmcs.Y_test,
                             data_amd.Y_test),
                            axis=0)

    logging.info('Size of X_Train: %s',X_train.shape)
    logging.info('Size of X_Train: %s', Y_train.shape)
    logging.info('Data loaded succesfully')

    mercury_model = M1Model(num_features)

    if load == 1:
        #load model from h5 file
        mercury_model.load(".\saved_models\\Mercury 1.h5")
        results = mercury_model.model.evaluate(X_test,Y_test,batch_size=32)
        print('test loss, test acc:',results)

    else:
        #build model
        print('Create the model.')
        mercury_model.build_model()

        #train model
        print('Create the trainer')
        trainer = Trainer(mercury_model.model,X_train,Y_train,epochs=10,batch_size=32)
        print('Start training the model.')
        trainer.train()

        #save model
        mercury_model.save(".\saved_models\\Mercury 1.h5")


if __name__ == '__main__':
    data_aapl = DataLoader(_AAPL,
                      window=10,
                      threshold=0.03,
                      technicals=0,
                      featselect=0,
                      drop=0)

    data_cmcs = DataLoader(_CMCSA,
                      window=10,
                      threshold=0.03,
                      technicals=0,
                      featselect=0,
                      drop=0)

    data_amd = DataLoader(_AMD,
                      window=10,
                      threshold=0.03,
                      technicals=0,
                      featselect=0,
                      drop=0)

    main(load=0)
    logging.info('Successful execution')
    os.system("tensorboard --logdir=.\\logs\\")

