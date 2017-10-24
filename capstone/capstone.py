import pandas as pd
import quandl
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression, BayesianRidge

from sklearn.metrics import accuracy_score, r2_score, explained_variance_score, mean_squared_error


def  Pred(code):
    quandl.ApiConfig.api_key = 'rTGnw5nnfs-ysvLvZ1fh'
    data = quandl.get(code, end_date="2016-09-24")
    test = quandl.get(code,  start_date="2016-09-24", end_date = "2017-09-24")
    
    data['HL_PCT'] = (data['High']-data['Low'])/data['Low'] * 100

    data['OC_PCT'] = (data['Open']-data['Close'])/data['Close'] * 100
    data = data[['Close','Open','HL_PCT','OC_PCT','High','Low']]

    data.fillna(-99999,inplace=True)


    X = data.drop('Close',1)

    X1 = data.drop('Open',1)
    X1 = X1.drop(X1.index[[len(X1)-1]])

    X2 = data.drop('High',1)
    X2 = X2.drop(X2.index[[len(X1)-1]])

    y = data[['Close']]
    y1 = data[['Open']]
    y1 = y1.drop(y1.index[[0]])

    y2 = data[['High']]
    y2 = y2.drop(y2.index[[0]])

    test['HL_PCT'] = (test['High']-test['Low'])/test['Low'] * 100

    test['OC_PCT'] = (test['Open']-test['Close'])/test['Close'] * 100
    test = test[['Close','Open','HL_PCT','OC_PCT','High','Low']]

    test.fillna(-99999,inplace=True)


    Xt = test.drop('Close',1)

    Xt1 = test.drop('Open',1)
    Xt1 = Xt1.drop(Xt1.index[[len(Xt1)-1]])

    Xt2 = test.drop('High',1)
    Xt2 = Xt2.drop(Xt2.index[[len(Xt1)-1]])

    yt = test[['Close']]
    yt1 = test[['Open']]
    yt1 = yt1.drop(yt1.index[[0]])

    yt2 = test[['High']]
    yt2 = yt2.drop(yt2.index[[0]])

    print("closing price dataset")
    print(X.head())
    print("\n")
    print("label")
    print("\n")
    print(y.head())
    print("\n")
    print("next day opening price dataset")
    print(X1.head())
    print("\n")
    print("label")
    print("\n")
    print(y1.head())
    print("\n")
    print("next day high price dataset")
    print(X2.head())
    print("\n")
    print("label")
    print("\n")
    print(y2.head())
    print("\n")
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

    X1_train, X1_test, y1_train, y1_test = cross_validation.train_test_split(X1, y1, test_size=0.2)

    X2_train, X2_test, y2_train, y2_test = cross_validation.train_test_split(X2, y2, test_size=0.2)

    
    clf = LinearRegression(n_jobs = -1)
    clf.fit(X_train,y_train)

    clf1 = LinearRegression(n_jobs = -1)
    clf1.fit(X1_train,y1_train)

    clf2 = LinearRegression(n_jobs = -1)
    clf2.fit(X2_train,y2_train)

    reg = linear_model.BayesianRidge()
    reg.fit(X_train,y_train)

    reg1 = linear_model.BayesianRidge()
    reg1.fit(X1_train,y1_train)

    reg2 = linear_model.BayesianRidge()
    reg2.fit(X2_train,y2_train)    

    LG = linear_model.Ridge()
    LG.fit(X_train,y_train)

    LG1 = linear_model.Ridge()
    LG1.fit(X1_train,y1_train)

    LG2 = linear_model.Ridge()
    LG2.fit(X2_train,y2_train)

    print("Here are the scores for the stock:{}".format(code))

    print("score for closing price prediction using linear regression \n")
    accuracy_LR = clf.score(X_test,y_test)
    print("accuracy is:{}".format(accuracy_LR))
    y_pred = clf.predict(X_test)
    print("r2_score is:{}".format(r2_score(y_test, y_pred)))
   

    print("score for next day opening price prediction using linear regression \n")

    accuracy_LR = clf1.score(X1_test,y1_test)
    print("accuracy is:{}".format(accuracy_LR))
    y1_pred = clf1.predict(X1_test)
    print("r2_score is:{}".format(r2_score(y1_test, y1_pred)))
   
    print("score for next day high price prediction using linear regression \n")

    accuracy_LR = clf2.score(X2_test,y2_test)
    print("accuracy is:{}".format(accuracy_LR))
    y2_pred = clf.predict(X2_test)
    print("r2_score is:{}".format(r2_score(y2_test, y2_pred)))
   
    print("score for closing price prediction using Bayesian Ridge regression \n")

    accuracy_LR = reg.score(X_test,y_test)
    print("accuracy is:{}".format(accuracy_LR))
    y_pred = clf.predict(X_test)
    print("r2_score is:{}".format(r2_score(y_test, y_pred)))
   

    print("score for next day opening price prediction using Bayesian Ridge regression \n")

    accuracy_LR = reg1.score(X1_test,y1_test)
    print("accuracy is:{}".format(accuracy_LR))
    y1_pred = reg1.predict(X1_test)
    print("r2_score is:{}".format(r2_score(y1_test, y1_pred)))
   

    print("score for next day high price prediction using Bayesian Ridge regression \n")

    accuracy_LR = reg1.score(X2_test,y2_test)
    print("accuracy is:{}".format(accuracy_LR))
    y2_pred = reg2.predict(X2_test)
    print("r2_score is:{}".format(r2_score(y2_test, y2_pred)))

    print("----------------------------------------------------------------")

    print("accuracy on validation set for linear regression model in predicting closing price is \n")

    accuracy_LR = clf.score(Xt,yt)
    print("accuracy is:{}".format(accuracy_LR))
    yt_pred = clf.predict(Xt)
    print("r2_score is:{}".format(r2_score(yt, yt_pred)))

    print("accuracy on validation set for linear regression model in predicting next day opening price is \n")

    accuracy_LR = clf1.score(Xt1,yt1)
    print("accuracy is:{}".format(accuracy_LR))
    yt1_pred = clf1.predict(Xt1)
    print("r2_score is:{}".format(r2_score(yt1, yt1_pred)))

    print("accuracy on validation set for linear regression model in predicting next day high price is \n")

    accuracy_LR = clf2.score(Xt2,yt2)
    print("accuracy is:{}".format(accuracy_LR))
    yt2_pred = clf2.predict(Xt2)
    print("r2_score is:{}".format(r2_score(yt2, yt2_pred)))

    print("accuracy on validation set for Bayesian Ridge regression model in predicting closing price is \n")

    accuracy_LR = reg.score(Xt,yt)
    print("accuracy is:{}".format(accuracy_LR))
    yt_pred_reg = reg.predict(Xt)
    print("r2_score is:{}".format(r2_score(yt, yt_pred)))

    print("accuracy on validation set for Bayesian Ridge regression model in predicting next day opening price is \n")

    accuracy_LR = reg1.score(Xt1,yt1)
    print("accuracy is:{}".format(accuracy_LR))
    yt1_pred_reg = reg1.predict(Xt1)
    print("r2_score is:{}".format(r2_score(yt1, yt1_pred)))

    print("accuracy on validation set for Bayesian Ridge regression model in predicting next day high price is \n")

    accuracy_LR = reg2.score(Xt2,yt2)
    print("accuracy is:{}".format(accuracy_LR))
    yt2_pred_reg = reg2.predict(Xt2)
    print("r2_score is:{}".format(r2_score(yt2, yt2_pred)))

    print("----------------------------------------------------------------")

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax1.plot(yt.index,yt,'r')
    ax2.plot(yt.index,yt_pred,'b')
    ax1.title.set_text('closing price actual')
    ax2.title.set_text('Closing price predicted by linear regression')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax1.plot(yt1.index,yt1,'g')
    ax2.plot(yt1.index,yt1_pred,'y')
    ax1.title.set_text('opening price actual')
    ax2.title.set_text('opening price predicted by linear regression')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax1.plot(yt2.index,yt2,'g')
    ax2.plot(yt2.index,yt2_pred,'y')
    ax1.title.set_text('High price actual')
    ax2.title.set_text('High price predicted by linear regression')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax1.plot(yt.index,yt,'r')
    ax2.plot(yt.index,yt_pred_reg,'b')
    ax1.title.set_text('closing price actual')
    ax2.title.set_text('Closing price predicted by Bayesian Ridge regression')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax1.plot(yt1.index,yt1,'g')
    ax2.plot(yt1.index,yt1_pred_reg,'y')
    ax1.title.set_text('opeing price actual')
    ax2.title.set_text('opening price predicted by Bayesian Ridge regression')
    plt.show()

    fig = plt.figure()
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222)
    ax1.plot(yt2.index,yt2,'g')
    ax2.plot(yt2.index,yt2_pred_reg,'y')
    ax1.title.set_text('High price actual')
    ax2.title.set_text('High price predicted by Bayesian Ridge regression')
    plt.show()

flag = 1

while flag:
    code = input("please input the quandl code for stock prediction in double quotes: or E in double quotes to exit")
    print(code)
    if code=="E":
        flag = 0
        
    else:
        print("default inputs being predicted")
        Pred(code)

code1 = "EOD/AAPL"
code2 = "EOD/BA"
code3 = "NSE/TATAMOTORS"

Pred(code1)
Pred(code2)
Pred(code3)

