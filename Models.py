import pandas as pd
import numpy as np
from sklearn import preprocessing, neural_network, metrics, tree, neighbors, svm, ensemble, model_selection
import matplotlib.pyplot as plt
data = pd.read_csv('sensor_readings_24.data.txt')
print data.shape
data = data.as_matrix()



train_X, test_X, train_Y, test_Y = model_selection.train_test_split(data[:, :-1], data[:, -1], test_size=0.3, random_state=39, stratify=data[:, -1])

scaler = preprocessing.StandardScaler().fit(train_X)
train_X = scaler.transform(train_X)
test_X = scaler.transform(test_X)


#Neural Network
NN = neural_network.MLPClassifier()
NN.fit(train_X, train_Y)

NN_predicted = NN.predict(test_X)

#Decision Tree
DT = tree.DecisionTreeClassifier()
DT.fit(train_X, train_Y)

DT_predicted = DT.predict(test_X)

#SVM
SupVec = svm.SVC(verbose=False)
SupVec.fit(train_X, train_Y)

svm_predicted = SupVec.predict(test_X)

#K-NN
KNN = neighbors.KNeighborsClassifier()
KNN.fit(train_X, train_Y)

KNN_predicted = KNN.predict(test_X)

#Boosting
Boost = ensemble.GradientBoostingClassifier(verbose=False)
Boost.fit(train_X, train_Y)

Boost_predicted = Boost.predict(test_X)


print "Accuracy by Model:_________________________________"
print 'Neural Net:{}'.format(metrics.accuracy_score(test_Y, NN_predicted))
print 'DT:{}'.format(metrics.confusion_matrix(test_Y, DT_predicted))
print 'SVM:{}'.format(metrics.accuracy_score(test_Y, svm_predicted))
print 'KNN:{}'.format(metrics.accuracy_score(test_Y, KNN_predicted))
print 'Boosting:{}'.format(metrics.accuracy_score(test_Y, Boost_predicted))
# print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_Y, KNN_predicted))

# a,tr,ts = model_selection.learning_curve(KNN,train_X,train_Y, cv=6)
#
# print tr.mean(axis=1)
# print a
#
# plt.plot(a, tr.mean(axis=1),'r-')
# plt.plot(a, ts.mean(axis=1),'b-')
# plt.show()