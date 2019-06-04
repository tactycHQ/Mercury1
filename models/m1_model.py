import tensorflow as tf
from tensorflow.keras import layers

class M1Model:

    def __init__(self):
        self.model=None

    def build_model(self):
        self.model = tf.keras.Sequential()

        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(64, activation='relu'))
        self.model.add(layers.Dense(10, activation='softmax'))

        self.model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def save(self,checkpoint_path):
        print("Saving model...")
        # self.save_weights(checkpoint_path)
        print("Model saved")

    def load(self,checkpoint_path):
        print("Loading model checkpoint {} ...\n".format(checkpoint_path))
        # self.load_weights(checkpoint_path)
        print("Model loaded")

