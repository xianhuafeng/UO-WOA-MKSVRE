from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import StackingRegressor
import time
from data_preprocession import *


def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))
def SSE(y_true, y_pred):
    return np.sum((y_true-y_pred)**2)
def SAE(y_true, y_pred):
    return np.sum(np.abs(y_true-y_pred))
def TIC(y_true, y_pred):
    a = np.sqrt(np.mean((y_true-y_pred)**2))
    b = np.sqrt(np.mean(y_true**2)) + np.sqrt(np.mean(y_pred**2))
    return a/b
def STD(y_true, y_pred):
    e = y_true-y_pred
    e_ = np.mean(e)
    sum_ = np.sum((e-e_)**2)
    return np.sqrt(sum_/(len(y_true)-1))
def VFE(y_true, y_pred):
    RE = y_true - y_pred
    return np.var(RE)
def DA(y_true, y_pred):
    w = np.zeros(len(y_true)-1)
    for i in range(len(y_true)-1):
        if (y_true[i+1]-y_true[i])*(y_pred[i+1]-y_true[i])>0:
            w[i] = 1
        else:
            w[i] = 0
    return np.mean(w)


def LSVR():
    linearSVR = SVR(kernel="linear")
    linearSVR.fit(x_train, y_train)
    preds = linearSVR.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))

def PSVR():
    polySVR = SVR(kernel="poly")
    polySVR.fit(x_train, y_train)
    preds = polySVR.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))

def RSVR():
    rbfSVR = SVR(kernel="rbf")
    rbfSVR.fit(x_train, y_train)
    preds = rbfSVR.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))

def SSVR():
    sigSVR = SVR(kernel="sigmoid")
    sigSVR.fit(x_train, y_train)
    preds = sigSVR.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))

def MKSVRE():
    estimators = [
        ("linear", SVR(kernel="linear")),
        ("poly", SVR(kernel="poly")),
        ("rbf", SVR(kernel="rbf")),
        ("sigmoid", SVR(kernel="sigmoid"))
    ]
    reg = StackingRegressor(estimators=estimators, final_estimator=LinearRegression(), cv=3, n_jobs=-1)
    reg.fit(x_train, y_train)
    preds = reg.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))



def main():
    print("-----------------------LSVR--------------------------")
    LSVR()
    print("-----------------------PSVR--------------------------")
    PSVR()
    print("-----------------------RSVR--------------------------")
    RSVR()
    print("-----------------------SSVR--------------------------")
    SSVR()
    print("-----------------------MKSVRE--------------------------")
    MKSVRE()

if __name__ == '__main__':
    x_train, y_train, x_test, y_test, y_scale = siteB_dataset()
    y_test = y_scale.inverse_transform(y_test)
    start = time.time()
    main()
    print(time.time()-start)