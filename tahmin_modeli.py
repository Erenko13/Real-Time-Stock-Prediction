import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import yfinance as yf
import math
from sklearn.preprocessing import MinMaxScaler
# %matplotlib inline


def train_datasi_yap(scaled_data, training_data_size):
    train_data = scaled_data[0:training_data_size, :]
    x_train = []
    y_train = []
    for i in range(60, len(train_data)):
        x_train.append(train_data[i-60:i, 0])
        y_train.append(train_data[i, 0])
        # if i < 61:
        #   print(x_train)
        #   print(y_train)
    return x_train, y_train


def test_datasi_yap(scaled_data, training_data_size, dataset):
    test_data = scaled_data[training_data_size - 60:, :]
    x_test = []
    y_test = dataset[training_data_size:, :]
    for i in range(60, len(test_data)):
        x_test.append(test_data[i - 60:i, 0])

    # x_test = np.array(x_test)
    # x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
    return x_test, y_test


def tahmin_yap(x_test, model, scaler):
    predictions = model.predict(x_test)
    an_array = predictions
    shape = np.shape(an_array)
    padded_array = np.zeros((len(predictions), 6))
    padded_array[:shape[0], :shape[1]] = an_array
    predictions = padded_array
    predictions = scaler.inverse_transform(predictions)[:, [0]]
    return predictions


def model_egit(ticker):
    df = yf.download(tickers=ticker, period='7d', interval='1m')
    df.head()

    print('eren')
    # preprocess başı
    data = df.filter(['Close'])
    dataset = df.values

    # %70 - %30 train-test ayırma
    training_data_size = math.ceil(len(dataset) * .7)

    # minmax scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(dataset)
    # train datası yap
    x_train, y_train = train_datasi_yap(scaled_data, training_data_size)

    # shapeleri ayarlama
    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    # model oluşturma
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    # model fiti
    model.fit(x_train, y_train, batch_size=1, epochs=1)
    return model, scaled_data, training_data_size, dataset, scaler, data

    # # rmse hesapla
    # rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
    # print(rmse)
    #
    # train = data[:training_data_size]
    # valid = data[training_data_size:]
    # valid['predictions'] = predictions
    # plt.figure(figsize=(16, 8))
    # plt.figure('Modle LM')
    # plt.xlabel('Tarih', fontsize=18)
    # plt.ylabel('Fiyat($)', fontsize=18)
    # plt.plot(train['Close'])
    # plt.plot(valid[['Close', 'predictions']])
    # plt.legend(['Train', 'Val', 'Predictions'], loc='upper left')
    # plt.show()


def train_wrapper(ticker):
    model, scaled_data, training_data_size, dataset, scaler, data = model_egit(ticker)
    return model, scaled_data, training_data_size, dataset, scaler, data


def test_wrapper(model, scaled_data, training_data_size, dataset, scaler, data):
    # test datası yap
    x_test, y_test = test_datasi_yap(scaled_data, training_data_size, dataset)
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # tahmin yap
    predictions = tahmin_yap(x_test, model, scaler)
    train = data[:training_data_size]
    valid = data[training_data_size:]
    valid['predictions'] = predictions
    return train,valid,predictions



# # rmse hesapla
# rmse = np.sqrt(np.mean(predictions - y_test) ** 2)
# print(rmse)
#
# train = data[:training_data_size]
# valid = data[training_data_size:]
# valid['predictions'] = predictions
# plt.figure(figsize=(16, 8))
# plt.figure('Modle LM')
# plt.xlabel('Tarih', fontsize=18)
# plt.ylabel('Fiyat($)', fontsize=18)
# plt.plot(train['Close'])
# plt.plot(valid[['Close', 'predictions']])
# plt.legend(['Train', 'Val', 'Predictions'], loc='upper left')
# plt.show()