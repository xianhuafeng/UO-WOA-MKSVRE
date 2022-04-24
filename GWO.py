import numpy as np
import random
import copy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D



''' Species initialization function '''
def initial(pop, dim, ub, lb):
    X = np.zeros([pop, dim])
    for i in range(pop):
        for j in range(dim):
            X[i, j] = random.random() * (ub[j] - lb[j]) + lb[j]

    return X, lb, ub



'''Boundary check function'''
def BorderCheck(X, ub, lb, pop, dim):
    for i in range(pop):
        for j in range(dim):
            if X[i, j] > ub[j]:
                X[i, j] = ub[j]
            elif X[i, j] < lb[j]:
                X[i, j] = lb[j]
    return X


'''Calculating the fitness function'''
def CaculateFitness(X, fun):
    pop = X.shape[0]
    fitness = np.zeros([pop, 1])
    for i in range(pop):
        fitness[i] = fun(X[i, :])
    return fitness



'''Fitness ranking'''
def SortFitness(Fit):
    fitness = np.sort(Fit, axis=0)
    index = np.argsort(Fit, axis=0)
    return fitness, index




'''Ranking of positions according to fitness'''
def SortPosition(X, index):
    Xnew = np.zeros(X.shape)
    for i in range(X.shape[0]):
        Xnew[i, :] = X[index[i], :]
    return Xnew




'''Gray Wolf Algorithm'''
def GWO(pop, dim, lb, ub, MaxIter, fun):
   
    Alpha_pos=np.zeros([1,dim])
    Alpha_score = float("inf")
    Beta_pos=np.ones([1,dim])
    Beta_score = float("inf")
    Delta_pos=np.ones([1,dim])
    Delta_score = float("inf")
    
    
    X, lb, ub = initial(pop, dim, ub, lb)
    fitness = CaculateFitness(X, fun)
    indexBest = np.argmin(fitness)
    GbestScore = copy.copy(fitness[indexBest])
    GbestPositon = np.zeros([1,dim])
    GbestPositon[0,:] = copy.copy(X[indexBest,:])
    Curve = np.zeros([MaxIter, 1])
    for t in range(MaxIter):
        
        for i in range(pop):            
            fitValue = fun(X[i,:])
            if fitValue<Alpha_score:
                Alpha_score = copy.copy(fitValue)
                Alpha_pos[0,:] = copy.copy(X[i,:])
            
            if fitValue>Alpha_score and fitValue<Beta_score:
                Beta_score = copy.copy(fitValue)
                Beta_pos[0,:] = copy.copy(X[i,:])
                
            if fitValue>Alpha_score and fitValue>Beta_score and fitValue<Delta_score:
                Delta_score = copy.copy(fitValue)
                Delta_pos[0,:] = copy.copy(X[i,:])
        
        a = 2 - t*(2/MaxIter)
        for i in range(pop):
            for j in range(dim):
                r1 = random.random()
                r2 = random.random()
                A1= 2*a*r1-a
                C1 = 2*r2
                
                D_alpha=np.abs(C1*Alpha_pos[0,j]-X[i,j])
                X1=Alpha_pos[0,j]-A1*D_alpha
                
                r1 = random.random()
                r2 = random.random()
                A2= 2*a*r1-a
                C2 = 2*r2
                
                D_beta=np.abs(C2*Beta_pos[0,j]-X[i,j])
                X2=Beta_pos[0,j]-A2*D_beta
                
                r1 = random.random()
                r2 = random.random()
                A3= 2*a*r1-a
                C3 = 2*r2
                D_beta=np.abs(C3*Delta_pos[0,j]-X[i,j])
                X3=Delta_pos[0,j]-A3*D_beta
                
                X[i,j] = (X1+X2+X3)/3
                
                  
        X = BorderCheck(X, ub, lb, pop, dim)
        fitness = CaculateFitness(X, fun)
        indexBest = np.argmin(fitness)
        if fitness[indexBest] <= GbestScore:
            GbestScore = copy.copy(fitness[indexBest])
            GbestPositon[0,:] = copy.copy(X[indexBest, :])
        Curve[t] = GbestScore

    return GbestScore, GbestPositon[0], Curve