# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 23:29:27 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

glass=pd.read_csv("glass.csv")

glass.shape

from sklearn.model_selection import train_test_split
train,test= train_test_split(glass,test_size=0.2, random_state=0)
from sklearn.neighbors import KNeighborsClassifier as KNC

acc=[]
for i in range(3,50,2):
    neigh=KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
    train_acc=np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9])
    test_acc= np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])
    acc.append([train_acc,test_acc])
    
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"bo-")
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"ro-")
plt.legend(["train","test"])

###consider knn=5

neigh=KNC(n_neighbors=5)
neigh.fit(train.iloc[:,0:9],train.iloc[:,9])
y_pred=neigh.predict(test.iloc[:,0:9])
y_pred

train_acc = np.mean(neigh.predict(train.iloc[:,0:9])==train.iloc[:,9]) 
train_acc  ###77.91%

test_acc=np.mean(neigh.predict(test.iloc[:,0:9])==test.iloc[:,9])
test_acc ##58.13%
