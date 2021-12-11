from tensorflow.keras import layers, Sequential
from tensorflow.keras.models import Model
import numpy as np


class AutoEncoders(Model):
    def __init__(self, input_dim,  architecture: list, name="autoencoder", **kwargs):
        super(AutoEncoders, self).__init__(name=name, **kwargs)
        self.input_dim = input_dim
        self.encoder = [layers.Dense(input_dim, activation="elu", input_shape=(input_dim,))] + [layers.Dense(dim, activation="elu") for dim in architecture[1:]]
        self.decoder = [layers.Dense(dim, activation="elu") for dim in architecture[::-1][1:]] + [layers.Dense(input_dim, activation="elu")]
        
    def call(self, x):
        # encoder
        for l in self.encoder:
            x = l(x)
        # decoder
        for l in self.decoder:
            x = l(x)
        return x
    
    def mad_score(self, points):
        """https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h.htm """
        m = np.median(points)
        ad = np.abs(points - m)
        mad = np.median(ad)

        return 0.6745 * ad / mad
    
    def get_outliers(self, x, threshold: float=5):
        reconstructions = self.predict(x)
        mse = np.mean(np.power(x - reconstructions, 2), axis=1)
        z_scores = self.mad_score(mse)
        return z_scores > threshold
