import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
import os
import numpy as np
from PIL import Image

class SRCNN:
    def __init__(self, lr_shape=(1024, 1024, 1)):
        self.model = self.build_model(lr_shape)

    def build_model(self, lr_shape):
        input_img = Input(shape=lr_shape)

        conv1 = Conv2D(64, (9, 9), activation='relu', padding='same')(input_img)
        conv2 = Conv2D(32, (1, 1), activation='relu', padding='same')(conv1)
        output_img = Conv2D(1, (5, 5), activation='linear', padding='same')(conv2)

        model = Model(input_img, output_img)
        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_squared_error'])
        return model

    def train(self, lr_images, hr_images, epochs=100, batch_size=4):
        return self.model.fit(lr_images, hr_images, epochs=epochs, batch_size=batch_size, validation_split=0.1)

    def predict(self, lr_image):
        return self.model.predict(lr_image)

    def save_model(self, file_path):
        self.model.save(file_path)

    def load_model(self, file_path):
        self.model = tf.keras.models.load_model(file_path)

    def evaluate(self, lr_images, hr_images):
        return self.model.evaluate(lr_images, hr_images)
