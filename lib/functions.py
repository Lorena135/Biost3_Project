import xgboost
from pathlib import Path
import urllib
import pandas as pd
import os
import numpy as np
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from numpy import *
import math
import matplotlib.pyplot as plt

######################FONCTIONS#########################


def precision(crosstab) :
    TP_BRCA = crosstab.iloc[0,0]
    TP_KIRC = crosstab.iloc[1,1]
    TP_COAD = crosstab.iloc[2,2]
    TP_LUAD = crosstab.iloc[3,3]
    TP_PRAD = crosstab.iloc[4,4]

    FP_BRCA = crosstab.iloc[1,0] + crosstab.iloc[2,0] + crosstab.iloc[3,0] + crosstab.iloc[4,0]
    FP_KIRC = crosstab.iloc[0,1] + crosstab.iloc[2,1] + crosstab.iloc[3,1] + crosstab.iloc[4,1]
    FP_COAD = crosstab.iloc[0,2] + crosstab.iloc[1,2] + crosstab.iloc[3,2] + crosstab.iloc[4,2]
    FP_LUAD = crosstab.iloc[0,3] + crosstab.iloc[1,3] + crosstab.iloc[2,3] + crosstab.iloc[4,3]
    FP_PRAD = crosstab.iloc[0,4] + crosstab.iloc[1,4] + crosstab.iloc[2,4] + crosstab.iloc[3,4]

    P_BRCA = TP_BRCA / (TP_BRCA + FP_BRCA)
    P_KIRC = TP_KIRC / (TP_KIRC + FP_KIRC)
    P_COAD = TP_COAD / (TP_COAD + FP_COAD)
    P_LUAD = TP_LUAD / (TP_LUAD + FP_LUAD)
    P_PRAD = TP_PRAD / (TP_PRAD + FP_PRAD)

    P_model = (P_BRCA + P_KIRC + P_COAD + P_LUAD + P_PRAD) / 5
    return P_model



####### Learning curve

def learning_curve(x_train, y_train, x_test, y_test, model) :
    testing = []
    training = []
    by_train = []
    for i in range(5, 115):
        model.fit(x_train[1:i], y_train[1:i], batch_size=32, epochs=100, verbose=1)
        by_train = np.argmax(y_train[1:i], axis=1)
        training_accuracy = accuracy_score(np.argmax(model.predict(x_train [1:i]), axis=1), by_train)
        by_test = np.argmax(y_test, axis=1)
        testing_accuracy = accuracy_score(np.argmax(model.predict(x_test), axis=1), by_test)
        training.append(training_accuracy)
        testing.append(testing_accuracy)

        print(training)
        print(testing)

    n = range(5,115)
    print(n)
    plt.plot(n,testing, "b" )
    plt.plot(n, training, "r")
    plt.show()

#### softmax function
def softmax(z):
    expz = np.exp(z)
    return expz / expz.sum()
