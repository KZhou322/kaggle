import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn import decomposition
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss
from sklearn.svm import SVC

trainFileFeature = pd.read_csv('train.csv',nrows=100)
print('1')
trainFileFeatureNoIP = trainFileFeature.drop(['ip','attributed_time','click_time','is_attributed'],axis=1)
print('2')
trainFileIP = trainFileFeature.drop(['app','device','os','channel','click_time','attributed_time','is_attributed'],axis=1)
print('3')
trainFileTruth = pd.read_csv('train.csv', usecols = [7],nrows=100)
print('4')
testFileFeature = pd.read_csv('test.csv')
print(testFileFeature.shape)
print('5')
testFileFeatureNoIP = testFileFeature.drop(['click_id','ip','click_time'], axis=1)
print('6')
testFileIP = testFileFeature.drop(['click_id','app','device','os','channel','click_time'],axis = 1)
print('7')
foroutput = pd.read_csv('test.csv', usecols = [0])
print(foroutput.head)
print('8')

def convertTime(dataframe):
    dataframe['sec'] = pd.to_datetime(dataframe.click_time).dataframe.second.astype('uint8')
    dataframe['min'] = pd.to_datetime(dataframe.click_time).dataframe.minute.astype('uint8')
    dataframe['hour'] = pd.to_datetime(dataframe.click_time).dataframe.hour.astype('uint8')
    dataframe.drop(['click_time'], axis=1, inplace=True)
print('9')
convertTime(trainFileFeatureNoIP)
print(trainFileFeatureNoIP.head())
print('10')
convertTime(testFileFeatureNoIP)
print('11')
ada = AdaBoostClassifier(n_estimators = 200)
#svc = SVC()
tree = DecisionTreeClassifier(random_state = 0)
print('12')
#svc.fit(trainFileIP,np.ravel(trainFileTruth))
ada.fit(trainFileFeatureNoIP,np.ravel(trainFileTruth))
print('13')
#predictthree = ada.predict(trainFileFeatureNoIP)
predictthreereal = ada.predict(testFileFeatureNoIP)
print(predictthreereal.shape)
print('14')
#predictfour = svc.predict(trainFileIP)
#predictfourreal = svc.predict(testFileIP)
print('15')
#combinePredictions = pd.DataFrame(predictfour,predictthree)
#combinePredictionsReal = pd.DataFrame(predictfourreal,predictthreereal)
print('16')
#tree.fit(combinePredictions,np.ravel(trainFileTruth))
print('17')
#combine = tree.predict(combinePredictionsReal)
print('18')
#out = np.asarray(combine)
predictfinal = pd.DataFrame(predictthreereal , columns = ['is_attributed'])
foroutput = foroutput.join(predictfinal)
#foroutput.rename(['click_id','is_attributed'],axis = 'columns')
print(foroutput.head)

foroutput.to_csv('result.csv',index = False)
