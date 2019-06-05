import logging
from data_loader.m1_data_loader import DataLoader
from models.m1_model import M1Model
from trainers.m1_trainer import M1ModelTrainer


#GLOBAL VARIABLES
__AAPL__ = "D:\\Dropbox\\9. Data\\Mercury Data\\CSV\\CIQ_AAPL_optimized.csv"


def main(load=0):

    X_train = data.X_train_std
    Y_train = data.Y_train
    features=X_train.shape[1]
    logging.info('Data loaded succesfully')

    m1_model = M1Model(features)

    if load == 1:
        #load model from h5 file
        m1_model.load("C:\\Users\\anubhav\\Desktop\\Projects\\Mercury1\\saved_models\\Mercury 1.h5")
    else:
        #build model
        print('Create the model.')
        m1_model.build_model()

        #train model
        print('Create the trainer')
        trainer = M1ModelTrainer(m1_model.model,X_train,Y_train,epochs=20,batch_size=32)
        print('Start training the model.')
        trainer.train()

        #save model
        m1_model.save("C:\\Users\\anubhav\\Desktop\\Projects\\Mercury1\\saved_models\\Mercury 1.h5")


if __name__ == '__main__':

    data = DataLoader(__AAPL__,
                      window=10,
                      threshold=0.03,
                      technicals=0,
                      featselect=0,
                      drop=0)

    main(load=0)
    logging.info('Successful execution')

