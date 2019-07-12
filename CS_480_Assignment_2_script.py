
import numpy as np
import sklearn.ensemble as learn
from sklearn.ensemble import AdaBoostRegressor
from skimage.feature import (daisy)
import csv 

#Load data sets
xtrain = np.loadtxt(open("MNIST_Xtrain.csv"),delimiter=",")
ytrain = np.loadtxt(open("MNIST_ytrain.csv"),delimiter=",")
xtest = np.loadtxt(open("MNIST_Xtestp.csv"),delimiter=",")

#generates CSV file in the required format for kaggle submission
def generateCSV(data):
    ofile  = open('shanth_output.csv',"wb")
    writer = csv.writer(ofile, delimiter=',')
    writer.writerow(['ImageID', 'Digit'])

    for i in range(data.shape[0]):
        writer.writerow(np.array([int(i+1),data[i]],dtype=np.uint64))
        

#Boosted trees with DAISY feature extraction
xtrain_daisy = np.zeros((xtrain.shape[0],3200))
for i in range(xtrain.shape[0]):
    img = xtrain[i,:].reshape((28,28),order='F')
    xtrain_daisy[i,:] = np.reshape(daisy(img,step=3,radius=8),3200)


xtest_daisy = np.zeros((xtest.shape[0],3200))
for i in range(xtest.shape[0]):
    img = xtest[i,:].reshape((28,28),order='F')
    xtest_daisy[i,:] = np.reshape(daisy(img,step=3,radius=8),3200)

randFor = learn.RandomForestClassifier(n_jobs=-1,n_estimators=15,max_depth=8)
bdt_discrete = AdaBoostRegressor(randFor,n_estimators=400)
bdt_discrete.fit(xtrain_daisy, ytrain)
ypredict = bdt_discrete.predict(xtest_daisy)

generateCSV(ypredict)

print 'Done'
