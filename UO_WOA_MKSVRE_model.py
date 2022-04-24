import numpy as np
import random
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import  cross_val_predict
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.metrics import mean_absolute_error as MAE
from sklearn.ensemble import StackingRegressor
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

'''Call this function to build the UO-WOA-MKSVRE model'''
def UO_WOA_MKSVRE_opt():
    def fitness_MKSVRE(X):
        base_models = [
            ("linear", SVR(kernel="linear", C=X[0], epsilon=X[1])),
            ("poly", SVR(kernel="poly", C=X[2], gamma=X[3], epsilon=X[4], degree=int(X[5]), coef0=X[6])),
            ("rbf", SVR(kernel="rbf", C=X[7], gamma=X[8], epsilon=X[9])),
            ("sigmoid", SVR(kernel="sigmoid", C=X[10], gamma=X[11], epsilon=X[12], coef0=X[13]))
        ]
        MKSVRE = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), cv=3, n_jobs=-1)
        preds = cross_val_predict(MKSVRE, x_train, y_train, cv=3, n_jobs=-1)
        return MSE(y_train, preds, squared=False)
    print("---------------------UO-WOA-MKSVRE model-------------------------")
    lb = [0.00001, 0, 0.00001, 0.00001, 0, 1, 0, 0.00001, 0.00001, 0, 0.00001, 0.000001, 0, 0]
    ub = [  10,   1,   100,     0.3,   1, 2, 5,   1000,      1,   1,  1000,       1,    1, 5]
    _, UO_WOA_MKSVRE_para, _  = WOA(pop=10,dim=len(lb),lb=lb,ub=ub,MaxIter=30,fun=fitness_MKSVRE)
    print("UO-WOA-MKSVRE model: " + str(UO_WOA_MKSVRE_para))
    base_models = [
        ("linear", SVR(kernel="linear", C=UO_WOA_MKSVRE_para[0], epsilon=UO_WOA_MKSVRE_para[1])),
        ("poly", SVR(kernel="poly", C=UO_WOA_MKSVRE_para[2], gamma=UO_WOA_MKSVRE_para[3], epsilon=UO_WOA_MKSVRE_para[4],
                     degree=int(UO_WOA_MKSVRE_para[5]), coef0=UO_WOA_MKSVRE_para[6])),
        ("rbf", SVR(kernel="rbf", C=UO_WOA_MKSVRE_para[7], gamma=UO_WOA_MKSVRE_para[8], epsilon=UO_WOA_MKSVRE_para[9])),
        ("sigmoid", SVR(kernel="sigmoid", C=UO_WOA_MKSVRE_para[10], gamma=UO_WOA_MKSVRE_para[11], epsilon=UO_WOA_MKSVRE_para[12],
                        coef0=UO_WOA_MKSVRE_para[13]))
    ]
    UO_WOA_MKSVRE = StackingRegressor(estimators=base_models, final_estimator=LinearRegression(), cv=3, n_jobs=-1)
    UO_WOA_MKSVRE.fit(x_train, y_train)
    preds = UO_WOA_MKSVRE.predict(x_test)
    preds = y_scale.inverse_transform(preds)
    print("MAE: ", round(MAE(y_test, preds), 4))
    print("RMSE: ", round(MSE(y_test, preds, squared=False), 4))
    print("MAPE: ", round(MAPE(y_test, preds), 4))
    print("SAE: ", round(SAE(y_test, preds), 4))
    print("STD: ", round(STD(y_test, preds), 4))



if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    x_train, y_train, x_test, y_test, y_scale = siteA_dataset() #or siteB_dataset()
    y_test = y_scale.inverse_transform(y_test)
    UO_WOA_MKSVRE_opt()
    plt.show()