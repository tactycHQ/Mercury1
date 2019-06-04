from data_loader.m1_data_loader import DataLoader
from models.m1_model import M1Model
#GLOBAL VARIABLES
__AAPL__ = "D:\\Dropbox\\9. Data\\Mercury Data\\CSV\\CIQ_AAPL.csv"


def main():

    print('Create the data generator.')
    df_aapl = DataLoader(__AAPL__, window=10, threshold=0.03)

    print('Create the model.')
    m1_model = M1Model()

    print('Create the trainer')
#    trainer = SimpleMnistModelTrainer(model.model, data_loader.get_train_data(), config)

    print('Start training the model.')
 #   trainer.train()


if __name__ == '__main__':
    main()
    print("Model Successful")
