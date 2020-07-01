# -*- coding: utf-8 -*-
"""
Created on Mon Jun 29 15:51:34 2020

@author: Varun
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
zoo=pd.read_csv("Zoo.csv")
zoo.shape
zoo.describe()
zoo.groupby('animal name').size()

from sklearn.model_selection import train_test_split
train,test=train_test_split(zoo,test_size=0.2, random_state=0)
from sklearn.neighbors import KNeighborsClassifier as KNC 

acc=[]
for i in range(3,20,2):
    neigh = KNC(n_neighbors=i)
    neigh.fit(train.iloc[:,1:18],train.iloc[:,0])
    train_acc = np.mean(neigh.predict(train.iloc[:,1:18])==train.iloc[:,0])
    test_acc = np.mean(neigh.predict(test.iloc[:,1:18])==test.iloc[:,0])
    acc.append([train_acc,test_acc])
###train accuracy plot    
plt.plot(np.arange(3,20,2),[i[0] for i in acc],"bo-")
###test accuracy plot
plt.plot(np.arange(3,20,2),[i[1] for i in acc],"ro-")
plt.legend(["train","test"])


###from graph, knn=3 has highest acc value. so k=3
acc
neigh=KNC(n_neighbors=3)
neigh.fit(train.iloc[:,1:18],train.iloc[:,0])
y_pred = neigh.predict(test.iloc[:,1:18])
y_pred
# train accuracy 
train_acc = np.mean(neigh.predict(train.iloc[:,1:18])==train.iloc[:,0]) 
train_acc #27.5%
# test accuracy
test_acc = np.mean(neigh.predict(test.iloc[:,1:18])==test.iloc[:,0]) 
test_acc  #4.76%