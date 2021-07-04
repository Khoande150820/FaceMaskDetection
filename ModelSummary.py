import cv2
import tensorflow as tf
import numpy as np
from tensorflow.keras import models
from tensorflow.keras.utils import plot_model
from keras_visualizer import visualizer

model = models.load_model("model")
model.summary()