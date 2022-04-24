import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


def series_to_features(data, lag, preday, n_out, time_features = False):
    df = pd.DataFrame(data.values.reshape(-1, 1))
    cols, names = list(), list()
    for i in range(preday, 0, -1):
        cols.append(df.shift(i * 24))
        names.append('var(t-{})'.format(i * 24))
    for i in range(lag, 0, -1):
        cols.append(df.shift(i))
        names.append('var(t-{})'.format(i))

    if time_features:
        cols.append(pd.DataFrame(data.index.hour))
        names.append("hour")
        cols.append(pd.DataFrame(data.index.dayofweek))
        names.append("dayofweek")
        cols.append(pd.DataFrame(data.index.day))
        names.append("day")
        cols.append(pd.DataFrame(data.index.month))
        names.append("month")

    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names.append('var(t)')
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    return agg


def siteA_dataset(lag = 12):
    data = pd.read_csv("./data/2011年9月风电场十分钟风速--山东省蓬莱风电厂.csv", index_col='PCTimeStamp', parse_dates=["PCTimeStamp"])
    site = data['WTG01_Ambient WindSpeed Avg. (1)']
    print("Number of null values: {}".format(np.sum(site.isnull())))
    if np.sum(site.isnull()) > 0:
        site.fillna(method='pad', inplace=True)
    dataset = series_to_features(site, lag=lag, preday=0, n_out=1)
    dataset.index = site.index
    dataset.dropna(inplace=True)
    train_data = dataset['2011-09-01': '2011-09-19']
    x_train = train_data.iloc[:, train_data.columns != 'var(t)'].values
    y_train = train_data['var(t)'].values.reshape(-1,1)

    test_data = dataset['2011-09-20']
    x_test = test_data.iloc[:, test_data.columns != 'var(t)'].values
    y_test = test_data['var(t)'].values.reshape(-1,1)

    x_scale = StandardScaler()
    x_scale.fit(x_train)
    x_train = x_scale.transform(x_train)
    x_test = x_scale.transform(x_test)
    y_scale = StandardScaler()
    y_scale.fit(y_train)
    y_train = y_scale.transform(y_train).ravel()
    y_test = y_scale.transform(y_test).ravel()
    return x_train, y_train, x_test, y_test, y_scale


def siteB_dataset(lag = 12):
    data = pd.read_csv("./data/2011年9月风电场十分钟风速--山东省蓬莱风电厂.csv", index_col='PCTimeStamp', parse_dates=["PCTimeStamp"])
    site = data['WTG11_Ambient WindSpeed Avg. (11)']
    print("Number of null values{}".format(np.sum(site.isnull())))
    if np.sum(site.isnull()) > 0:
        site.fillna(method='pad', inplace=True)
    dataset = series_to_features(site, lag=lag, preday=0, n_out=1)
    dataset.index = site.index
    dataset.dropna(inplace=True)
    train_data = dataset['2011-09-01': '2011-09-19']
    x_train = train_data.iloc[:, train_data.columns != 'var(t)'].values
    y_train = train_data['var(t)'].values.reshape(-1,1)

    test_data = dataset['2011-09-20']
    x_test = test_data.iloc[:, test_data.columns != 'var(t)'].values
    y_test = test_data['var(t)'].values.reshape(-1,1)

    x_scale = StandardScaler()
    x_scale.fit(x_train)
    x_train = x_scale.transform(x_train)
    x_test = x_scale.transform(x_test)
    y_scale = StandardScaler()
    y_scale.fit(y_train)
    y_train = y_scale.transform(y_train).ravel()
    y_test = y_scale.transform(y_test).ravel()
    return x_train, y_train, x_test, y_test, y_scale

if __name__ == '__main__':
    x_train, y_train, x_test, y_test, y_scale = siteB_dataset()
    plt.show()
