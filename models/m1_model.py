import logging
import tensorflow as tf
from keras import layers, Sequential, optimizers, models

class M1Model:

    def __init__(self,features):
        self.model=None
        self.features=features
        tf.logging.set_verbosity(tf.logging.ERROR)

    def build_model(self):
        logging.info('Building model...')
        self.model = Sequential()

        self.model.add(layers.Dense(128, activation='relu',input_shape=(self.features,)))
        self.model.add(layers.Dense(128, activation='relu'))
        self.model.add(layers.Dense(3, activation='softmax'))

        logging.info('Compiling model...')
        self.model.compile(optimizer=optimizers.Adam(0.001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        print(self.model.summary())
        return self.model

    def save(self,checkpoint_path):
        logging.info("Saving model...")
        self.model.save(checkpoint_path)
        logging.info("Model saved")

    def load(self,checkpoint_path):
        logging.info("Loading model checkpoint {} ...\n".format(checkpoint_path))
        self.model= models.load_model(checkpoint_path)
        logging.info('Model loaded')

