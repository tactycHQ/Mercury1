from data_loader import m1_data_loader
import tensorflow as tf
from tensorflow.keras import layers

#GLOBAL VARIABLES
__AAPL__ = "D:\\Dropbox\\9. Data\\Mercury Data\\CSV\\CIQ_AAPL.csv"

df_aapl = m1_data_loader.DataLoader(__AAPL__, window=10, threshold=0.03)




print(tf.keras.__version__)