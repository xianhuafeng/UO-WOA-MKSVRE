import numpy as np
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.ensemble import StackingRegressor
import random
np.set_printoptions(suppress=True)
from data_preprocession import *
from WOA import WOA


def SSE(y_true, y_pred):
    return np.sum((y_true-y_pred)**2)
def SAE(y_true, y_pred):
    return np.sum(np.abs(y_true-y_pred))
def STD(y_true, y_pred):
    e = y_true-y_pred
    e_ = np.mean(e)
    sum_ = np.sum((e-e_)**2)
    return np.sqrt(sum_/(len(y_true)-1))

'''Call this function to build the IO-WOA-MKSVRE model'''
def IO_WOA_MKSVRE_opt():
    def fitness_LSVR(X):
        linearSVR = SVR(kernel="linear", C=X[0], epsilon=X[1])
        preds = cross_val_predict(linearSVR, x_train, y_train, cv=3, n_jobs=-1)
        return MSE(y_train, preds, squared=False)

    def fitness_PSVR(X):
        polySVR = SVR(kernel="poly", C=X[0], gamma=X[1], epsilon=X[2], degree=int(X[3]), coef0=X[4])
        preds = cross_val_predict(polySVR, x_train, y_train, cv=3, n_jobs=-1)
        return MSE(y_train, preds, squared=False)

    def fitness_RSVR(X):
        rbfSVR = SVR(kernel="rbf", C=X[0], gamma=X[1], epsilon=X[2])
        preds = cross_val_predict(rbfSVR, x_train, y_train, cv=3, n_jobs=-1)
        return MSE(y_train, preds, squared=False)

    def fitness_SSVR(X):
        sigSVR = SVR(kernel="sigmoid", C=X[0], gamma=X[1], epsilon=X[2], coef0=X[3])
        preds = cross_val_predict(sigSVR, x_train, y_train, cv=3, n_jobs=-1)
        return MSE(y_train, preds, squared=False)

    print("---------------------WOA-LSVR model-------------------------")
    lb = [0.00001, 0]
    ub = [10, 1]
    _, WOA_LSVR_para, _  = WOA(pop=10,dim=len(lb),lb=lb,ub=ub,MaxIter=30,fun=fitness_LSVR)
    print("WOA-LSVR model: " + str(WOA_LSVR_para))
    WOA_LSVR = SVR(kernel="linear", C=WOA_LSVR_para[0], epsilon=WOA_LSVR_para[1])
    WOA_LSVR.fit(x_train, y_train)
    preds = WOA_LSVR.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))

    print("---------------------WOA-PSVR model-------------------------")
    lb = [0.00001, 0.00001, 0, 1, 0]
    ub = [100, 0.3, 1, 2, 5]
    _, WOA_PSVR_para, _  = WOA(pop=10,dim=len(lb),lb=lb,ub=ub,MaxIter=30,fun=fitness_PSVR)
    print("WOA-PSVR model: " + str(WOA_PSVR_para))
    WOA_PSVR = SVR(kernel="poly", C=WOA_PSVR_para[0], gamma=WOA_PSVR_para[1], epsilon=WOA_PSVR_para[2],
                 degree=int(WOA_PSVR_para[3]), coef0=WOA_PSVR_para[4])
    WOA_PSVR.fit(x_train, y_train)
    preds = WOA_PSVR.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))

    print("---------------------WOA-RSVR model-------------------------")
    lb = [0.00001, 0.00001, 0]
    ub = [1000, 1, 1]
    _, WOA_RSVR_para, _  = WOA(pop=10,dim=len(lb),lb=lb,ub=ub,MaxIter=30,fun=fitness_RSVR)
    print("WOA-RSVR model: " + str(WOA_RSVR_para))
    WOA_RSVR = SVR(kernel="rbf", C=WOA_RSVR_para[0], gamma=WOA_RSVR_para[1], epsilon=WOA_RSVR_para[2])
    WOA_RSVR.fit(x_train, y_train)
    preds = WOA_RSVR.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))

    print("---------------------WOA-SSVR model-------------------------")
    lb = [0.00001, 0.000001, 0, 0]
    ub = [1000, 1, 1, 5]
    _, WOA_SSVR_para, _  = WOA(pop=10,dim=len(lb),lb=lb,ub=ub,MaxIter=30,fun=fitness_SSVR)
    print("WOA-SSV model: " + str(WOA_SSVR_para))
    WOA_SSVR = SVR(kernel="sigmoid", C=WOA_SSVR_para[0], gamma=WOA_SSVR_para[1], epsilon=WOA_SSVR_para[2], coef0=WOA_SSVR_para[3])
    WOA_SSVR.fit(x_train, y_train)
    preds = WOA_SSVR.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))

    print("---------------------IO-WOA-MKSVRE model-------------------------")
    IO_WOA_MKSVRE_para = WOA_LSVR_para.tolist() + WOA_PSVR_para.tolist() + WOA_RSVR_para.tolist() + WOA_SSVR_para.tolist()
    print("IO-WOA-MKSVRE model: " + str(IO_WOA_MKSVRE_para))
    base_models = [
        ("linear", SVR(kernel="linear", C=IO_WOA_MKSVRE_para[0], epsilon=IO_WOA_MKSVRE_para[1])),
        ("poly", SVR(kernel="poly", C=IO_WOA_MKSVRE_para[2], gamma=IO_WOA_MKSVRE_para[3], epsilon=IO_WOA_MKSVRE_para[4],
                     degree=int(IO_WOA_MKSVRE_para[5]), coef0=IO_WOA_MKSVRE_para[6])),
        ("rbf", SVR(kernel="rbf", C=IO_WOA_MKSVRE_para[7], gamma=IO_WOA_MKSVRE_para[8], epsilon=IO_WOA_MKSVRE_para[9])),
        ("sigmoid", SVR(kernel="sigmoid", C=IO_WOA_MKSVRE_para[10], gamma=IO_WOA_MKSVRE_para[11], epsilon=IO_WOA_MKSVRE_para[12],
                        coef0=IO_WOA_MKSVRE_para[13]))
    ]
    IO_WOA_MKSVRE = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), cv=3, n_jobs=-1)
    IO_WOA_MKSVRE.fit(x_train, y_train)
    preds = IO_WOA_MKSVRE.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))



if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    x_train, y_train, x_test, y_test, y_scale = siteA_dataset()
    y_test = y_scale.inverse_transform(y_test)
    IO_WOA_MKSVRE_opt()
    plt.show()