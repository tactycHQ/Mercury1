import tensorflow as tf
from keras import layers, Sequential, optimizers

class M1Model:

    def __init__(self):
        self.model=None

    def build_model(self):
        # print(tf.VERSION)
        print(tf.keras.__version__)

        self.model = Sequential()

        self.model.add(layers.Dense(64, activation='relu',input_shape=(137,)))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(3, activation='softmax'))


        self.model.compile(optimizer=optimizers.Adam(0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return self.model

    def save(self,checkpoint_path):
        print("Saving model...")
        # self.save_weights(checkpoint_path)
        print("Model saved")

    def load(self,checkpoint_path):
        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        # self.load_weights(checkpoint_path)
        print("Model loaded")

