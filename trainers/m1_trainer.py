from keras.callbacks import ModelCheckpoint, TensorBoard

class M1ModelTrainer:
    def __init__(self,model,X_train,Y_train,epochs,batch_size):
        self.model=model
        self.X_train=X_train
        self.Y_train=Y_train
        self.epochs=epochs
        self.batch_size=batch_size
        self.callbacks=[]
        self.loss=[]
        self.acc=[]
        self.val_loss=[]
        self.val_acc=[]

    def train(self):
        history = self.model.fit(
        self.X_train,
        self.Y_train,
        epochs=self.epochs,
        batch_size=self.batch_size,
        steps_per_epoch=None
        )