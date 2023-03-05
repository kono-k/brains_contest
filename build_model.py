import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import r2_score
from sklearn.base import BaseEstimator, ClassifierMixin
from matplotlib import pyplot as plt
import pandas as pd


class MyNetWork(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def build_model(self, X, y):
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=[len(X[0])]),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dropout(0.1),
            keras.layers.Dense(1)
        ])

        optimizer = tf.keras.optimizers.Adam(0.001)

        model.compile(loss='mse',
                      optimizer=optimizer,
                      metrics=['mae', 'mse'])

        return model

    def fit(self, X, y):
        X = np.array(X, dtype="float64")
        y = np.array(y, dtype="float64")

        class PrintDot(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs):
                if epoch % 100 == 0:
                    print('')
                print('.', end='')

        model = self.build_model(X, y)
        print(model.summary())
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=30)
        history = model.fit(X, y, epochs=1000, validation_split=0.2,
                            verbose=0, callbacks=[PrintDot(), early_stop])
        self.history = history
        self.model = model

        return self

    def predict(self, X):
        X = np.array(X, dtype="float64")
        model = self.model
        pred_list = model.predict(X, verbose=0)

        return pred_list

    def score(self, X, y):
        X = np.array(X, dtype="float64")
        y = np.array(y, dtype="float64")

        pred_list = self.predict(X)
        r2 = r2_score(y, pred_list)

        return r2
    
    def save(self, path):
        self.model.save(path)

    def plot_history(self):
        history = self.history
        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(hist['epoch'], hist['mae'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mae'],
                 label='Val Error')
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(hist['epoch'], hist['mse'],
                 label='Train Error')
        plt.plot(hist['epoch'], hist['val_mse'],
                 label='Val Error')
        plt.legend()
        plt.show()
