from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from data_preprocession import *
import random
np.set_printoptions(suppress=True)
from WOA import WOA
from GWO import GWO

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


def GWO_RF():
    def fitness_RF(X):
        rf = RandomForestRegressor(n_estimators=int(X[0]), max_depth=int(X[1]), max_features=int(X[2]), random_state=0,n_jobs=-1)
        preds = cross_val_predict(rf, x_train, y_train, cv=3, n_jobs=-1)
        return MSE(y_train, preds, squared=False)
    print("---------------------GWO-RF model-------------------------")
    lb = [10, 10, 1]
    ub = [200, 25, 12]
    _, GWO_RF_para, _ = GWO(pop=10, dim=len(lb), lb=lb, ub=ub, MaxIter=30, fun=fitness_RF)
    print("GWO-RF model: " + str(GWO_RF_para))
    GWO_RF = RandomForestRegressor(n_estimators=int(GWO_RF_para[0]), max_depth=int(GWO_RF_para[1]), max_features=int(GWO_RF_para[2]),random_state=0, n_jobs=-1)
    GWO_RF.fit(x_train, y_train)
    preds = GWO_RF.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))

def WOA_RF():
    def fitness_RF(X):
        rf = RandomForestRegressor(n_estimators=int(X[0]), max_depth=int(X[1]), max_features=int(X[2]), random_state=0,n_jobs=-1)
        preds = cross_val_predict(rf, x_train, y_train, cv=3, n_jobs=-1)
        return MSE(y_train, preds, squared=False)
    print("---------------------WOA-RF model-------------------------")
    lb = [10, 10, 1]
    ub = [200, 25, 12]
    _, WOA_RF_para, _ = WOA(pop=10, dim=len(lb), lb=lb, ub=ub, MaxIter=30, fun=fitness_RF)
    print("WOA-RF model: " + str(WOA_RF_para))
    WOA_RF = RandomForestRegressor(n_estimators=int(WOA_RF_para[0]), max_depth=int(WOA_RF_para[1]), max_features=int(WOA_RF_para[2]),random_state=0, n_jobs=-1)
    WOA_RF.fit(x_train, y_train)
    preds = WOA_RF.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))


def WOA_XGBoost():
    def fitness_XGB(X):
        xgb = XGBRegressor(n_estimators=int(X[0]), learning_rate=X[1], max_depth=int(X[2]),reg_alpha=X[3], reg_lambda=X[4],
                           random_state=0, n_jobs=-1)
        preds = cross_val_predict(xgb, x_train, y_train, cv=3, n_jobs=-1)
        return MSE(y_train, preds, squared=False)
    print("---------------------WOA-XGB model-------------------------")
    lb = [10, 0.001, 5,  0, 0]
    ub = [200, 1, 15,  5, 5]
    _, WOA_XGB_para, _ = WOA(pop=10, dim=len(lb), lb=lb, ub=ub, MaxIter=30, fun=fitness_XGB)
    print("WOA-XGB model: " + str(WOA_XGB_para))
    WOA_XGB = XGBRegressor(n_estimators=int(WOA_XGB_para[0]), learning_rate=WOA_XGB_para[1], max_depth=int(WOA_XGB_para[2]),reg_alpha=WOA_XGB_para[3],
                           reg_lambda=WOA_XGB_para[4], random_state=0, n_jobs=-1)
    WOA_XGB.fit(x_train, y_train)
    preds = WOA_XGB.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))


def XGBoost():
    xgboost = XGBRegressor(random_state=0, n_jobs=-1)
    xgboost.fit(x_train, y_train)
    preds = xgboost.predict(x_test)
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
    XGBoost()
