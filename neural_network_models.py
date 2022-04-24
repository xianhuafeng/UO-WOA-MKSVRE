from tensorflow import keras
from tensorflow.keras.layers import Dense,LSTM,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.model_selection import KFold
import random
from WOA import WOA
from data_preprocession import *


def SSE(y_true, y_pred):
    return np.sum((y_true-y_pred)**2)
def SAE(y_true, y_pred):
    return np.sum(np.abs(y_true-y_pred))
def STD(y_true, y_pred):
    e = y_true-y_pred
    e_ = np.mean(e)
    sum_ = np.sum((e-e_)**2)
    return np.sqrt(sum_/(len(y_true)-1))




def bpnn_cross_val_predict(X, x_train, y_train, cv):
    n_splits = cv
    oof_train = np.zeros(x_train.shape[0])
    for train_index, valid_index in KFold(n_splits).split(x_train):
        kf_x_train = x_train[train_index]
        kf_y_train = y_train[train_index]
        kf_x_valid = x_train[valid_index]
        model = keras.models.Sequential([
            keras.layers.Dense(units=int(X[0]), activation='relu', input_shape=(kf_x_train.shape[1],)),
            keras.layers.Dense(units=int(X[1]), activation='relu'),
            keras.layers.Dense(units=1)
        ])
        optimizer = Adam(learning_rate=X[2])
        model.compile(optimizer=optimizer, loss='mae')
        model.fit(kf_x_train, kf_y_train, epochs=10, batch_size=128)
        oof_train[valid_index] = model.predict(kf_x_valid).ravel()
    return oof_train

'''Back propagation neural network'''
def WOA_BPNN():
    x_train, y_train, x_test, y_test, y_scale = siteA_dataset()
    y_test = y_scale.inverse_transform(y_test)
    def fitness_bpnn(X):
        preds = bpnn_cross_val_predict(X, x_train, y_train, cv=3)
        return MSE(y_train, preds, squared=False)
    print("---------------------WOA-BPNN model-------------------------")
    lb = [1, 1, 0.00001]
    ub = [50, 50, 1]
    _, WOA_BPNN_para, _ = WOA(pop=10, dim=len(lb), lb=lb, ub=ub, MaxIter=30, fun=fitness_bpnn)
    print("WOA-BPNN model: " + str(WOA_BPNN_para))
    model = keras.models.Sequential([
        keras.layers.Dense(units=int(WOA_BPNN_para[0]), activation='relu', input_shape=(x_train.shape[1],)),
        keras.layers.Dense(units=int(WOA_BPNN_para[1]), activation='relu'),
        keras.layers.Dense(units=1)
    ])
    optimizer = Adam(learning_rate=WOA_BPNN_para[2])
    model.compile(optimizer=optimizer, loss='mae')
    model.fit(x_train, y_train, epochs=10, batch_size=128)
    preds = model.predict(x_test).ravel()
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))
    model.save('bpnn.h5')


def lstm_cross_val_predict(X, x_train, y_train, cv):
    n_splits = cv
    oof_train = np.zeros(x_train.shape[0])
    for train_index, valid_index in KFold(n_splits).split(x_train):
        kf_x_train = x_train[train_index]
        kf_y_train = y_train[train_index]
        kf_x_valid = x_train[valid_index]
        kf_x_train = kf_x_train.reshape((kf_x_train.shape[0], kf_x_train.shape[1], 1))
        kf_x_valid = kf_x_valid.reshape((kf_x_valid.shape[0], kf_x_valid.shape[1], 1))
        model = Sequential()
        model.add(LSTM(int(X[0]), activation='relu', return_sequences=True, input_shape=(kf_x_train.shape[1], 1)))
        model.add(LSTM(int(X[1]), activation='relu'))
        model.add(Dense(1))
        optimizer = Adam(learning_rate=X[2])
        model.compile(optimizer=optimizer, loss='mae')
        model.fit(kf_x_train, kf_y_train, epochs=10, batch_size=128)
        oof_train[valid_index] = model.predict(kf_x_valid).ravel()
    return oof_train

'''Long short-term memory'''
def WOA_LSTM():
    x_train, y_train, x_test, y_test, y_scale = siteA_dataset()
    y_test = y_scale.inverse_transform(y_test)
    def fitness_lstm(X):
        preds = lstm_cross_val_predict(X, x_train, y_train, cv=3)
        return MSE(y_train, preds, squared=False)
    print("---------------------WOA-LSTM model-------------------------")
    lb = [1, 1, 0.00001]
    ub = [30, 30, 1]
    _, WOA_LSTM_para, _ = WOA(pop=10, dim=len(lb), lb=lb, ub=ub, MaxIter=30, fun=fitness_lstm)
    print("WOA-LSTM model: " + str(WOA_LSTM_para))
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    model = Sequential()
    model.add(LSTM(int(WOA_LSTM_para[0]), activation='relu',return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(LSTM(int(WOA_LSTM_para[1]),activation='relu'))
    model.add(Dense(1))
    optimizer = Adam(learning_rate=WOA_LSTM_para[2])
    model.compile(optimizer=optimizer, loss='mae')
    model.fit(x_train, y_train, epochs=10, batch_size=128)
    preds = model.predict(x_test).ravel()
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))
    model.save('lstm.h5')



if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    WOA_BPNN()
