from data_loader.m1_data_loader import DataLoader
from models.m1_model import M1Model
from trainers.m1_trainer import M1ModelTrainer


#GLOBAL VARIABLES
__AAPL__ = "D:\\Dropbox\\9. Data\\Mercury Data\\CSV\\CIQ_AAPL.csv"


def main():

    print('Create the data generator.')
    df_aapl = DataLoader(__AAPL__, window=10, threshold=0.03)
    X_train = df_aapl.X_train_std
    Y_train = df_aapl.Y_train
    print(X_train.shape)
    print(Y_train.shape)

    print('Create the model.')
    m1_model = M1Model()
    m1_model.build_model()

    print('Create the trainer')
    trainer = M1ModelTrainer(m1_model.model,X_train,Y_train,epochs=20,batch_size=32)

    print('Start training the model.')
    trainer.train()


if __name__ == '__main__':
    main()
    print("Model Successful")
