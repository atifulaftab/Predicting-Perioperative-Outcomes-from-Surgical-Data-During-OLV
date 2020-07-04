# -*- coding: utf-8 -*-
"""
Created on Wed Dec 18 00:38:38 2019

@author: atifu
"""
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
#import fileReading as fr
import csv
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
import warnings
from sklearn.metrics import confusion_matrix
import pickle
from sklearn import metrics
import pandas as pd
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import f1_score
from sklearn.decomposition import PCA
M=10 #No of iteration
CV=5 #No of cross validation
recall = 0 #sensitivity
fallout = 0
precision = 0 
specificity = 0
f1 = 0
mcc = 0
auc = 0 

def plot_pca_svm(X,y,clf):

   # X is the input matrix

   # y is the output vector

   # clf is the trained classifier.

   pca = PCA(n_components=2)

   pca.fit(X)

   X_pca = pca.transform(X)

   x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1

   y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1

   h = 0.2

   xx, yy = np.meshgrid(np.arange(x_min, x_max, h),

                     np.arange(y_min, y_max, h))

 

   gg = np.c_[xx.ravel(),yy.ravel()]

   to_pred = pca.inverse_transform(gg)

   Z = clf.predict(to_pred)

   ZZ = Z.reshape(xx.shape) 

   

   plt.figure(figsize=(8,8))

   plt.contourf(xx,yy,ZZ, cmap=plt.cm.coolwarm, alpha=0.8)

   plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)

   plt.show()


def classification_report_csv(report):
    report_data = []
    lines = report.split('\t')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row['class'])
        report_data.append(row['precision'])
        report_data.append(row['recall'])
        report_data.append(row['f1_score'])
        report_data.append(row['support'])
        print(report_data)
    dataframe = pd.DataFrame.from_dict(report_data)
    #print(dataframe)
    dataframe.to_csv('classification_report.csv', index = False)


def readFile(markerfilename):
    data = []
    with open(markerfilename, 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            data.append(row)

    data = np.array(data)
    
    
    return data


def result_rf( X_train, y_train ,X_test, y_test) :
 classifier = RandomForestClassifier()
 length = M
 cross_validation = CV
 acc_score = np.empty(length)
 depth = np.empty(length)
 n_estimator = np.empty(length)
 acc_mat = np.empty(length)
 f1 = np.empty(length)
 f1_avg = np.empty(length)
 std= np.empty(length)
 std_avg = np.empty(length)
 featuresum = np.empty(16)
 for j in range(length):
  grid_param = {
      'n_estimators': [1,2,3,4,5,6,7,8,9,10,25,100,200],
      'max_depth' : [1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,25]
  }

  gd_sr = GridSearchCV(estimator=classifier,
                       param_grid=grid_param,
                       scoring='accuracy',
                       cv=cross_validation,
                       n_jobs=-1)
  gd_sr.fit(X_train, y_train)
  #print("Grid Search Best Score:", gd_sr.best_score_)
  warnings.filterwarnings('ignore')
  warnings.filterwarnings(action='ignore',category=DeprecationWarning)
  warnings.filterwarnings(action='ignore',category=FutureWarning)
  best_parameters = gd_sr.best_params_
  depth[j] = best_parameters['max_depth']
  n_estimator[j] = best_parameters['n_estimators']
  
  
      
  for i in range(length):
    classifier = RandomForestClassifier(n_estimators=best_parameters['n_estimators'],  max_depth=best_parameters['max_depth'])
    classifier.fit(X_train, y_train)
    filename = 'finalized_model.sav'
    pickle.dump(classifier, open(filename, 'wb'))
#    importances = classifier.feature_importances_
#    print(importances)
#    print("Iteration No:",i)
#    print("\n")
#     for feature in zip(classifier.feature_importances_):
#         feature=feature
# #        print(feature)    
#         featuresum[i]=feature
    

    y_pred = classifier.predict(X_test)
    acc_mat[i] = (accuracy_score(y_test, y_pred))
    f1[i]=f1_score(y_test, y_pred, average='weighted')
    std[i]=np.std(f1)
  importances = classifier.feature_importances_
  std = np.std([tree.feature_importances_ for tree in classifier.estimators_],
             axis=0)
  indices = np.argsort(importances)[::-1]

     # Print the feature ranking
  #print("Feature ranking:")

  for f in range(X.shape[1]):
#      print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
      featuresum[indices[f]]= featuresum[indices[f]]+importances[indices[f]]
  acc_score[j] = acc_mat.mean()
  f1_avg[j] = f1.mean()
  std_avg[j] = std.mean()
 confMatrix = confusion_matrix(y_test, y_pred)
# print(confMatrix)
# print('Cross Validation Scores',cross_val_score(classifier, X, y, cv=5))
# print('Mean CV Score',np.mean(cross_val_score(classifier, X, y, cv=5)))
 # tp=confMatrix[0,0]
 # fp=confMatrix[0,1]
 # fn=confMatrix[1,0]
 # tn=confMatrix[1,1]
 # recall = (tp/(tp+fn)) #sensitivity tpr
 # print('recall or sesitivity:', recall)
 # fallout = (fp/(fp+tn)) #fpr
 # print('fallout:',fallout)
 # precision = (tp/(tp+fp))
 # print('precision:',precision)
 # specificity = 1 - fallout #tnr
 # print('specificity:',specificity)
 # f1 = ((2*tp)/(2*tp+fp+fn))
 # print('f1:',f1)
 # mcc = ((tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)
 # print('mcc:',mcc)
 # auc = ((recall+specificity)/2)
 # print('auc:',auc)
 
 loaded_model = pickle.load(open(filename, 'rb'))
 result = loaded_model.score(X_test, y_test)
 #print("Dump Result",result)
 
# print(metrics.classification_report(y_test, y_pred))
 target_names = ['class 0', 'class 1']
 report = metrics.classification_report(y_test, y_pred, target_names=target_names)
 classification_report_csv(report)
 ax = plt.gca()
 rfc_disp = plot_roc_curve(classifier, X_test, y_test, ax=ax, alpha=0.8)
 plt.show()
 for f in range(X.shape[1]):
      print("%d. feature %d (%f)" % (f + 1, indices[f], featuresum[indices[f]]/length))
 return [acc_score,f1_avg, std_avg, depth,n_estimator]


def result_svm( X_train, y_train ,X_test, y_test) :
 classifier = SVC()
 length = M
 cross_validation = CV
 acc_score = np.empty(length)
 C = np.empty(length)
 gamma = np.empty(length)
 acc_mat = np.empty(length)
 f1 = np.empty(length)
 f1_avg = np.empty(length)
 std= np.empty(length)
 std_avg = np.empty(length)
 for j in range(length):
  grid_param = {
       'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000], 
       'gamma': [0.01, 0.001, 0.0001], 
       'kernel': ['rbf','linear']
  }
  gd_sr = GridSearchCV(estimator=classifier,
                       param_grid=grid_param,
                       scoring='accuracy',
                       cv=cross_validation,
                       n_jobs=-1)
  gd_sr.fit(X_train, y_train)
#  print("Grid Search Best Score:", gd_sr.best_score_)
  warnings.filterwarnings('ignore')
  warnings.filterwarnings(action='ignore',category=DeprecationWarning)
  warnings.filterwarnings(action='ignore',category=FutureWarning)
  best_parameters = gd_sr.best_params_
  C[j] = best_parameters['C']
  kernel = best_parameters['kernel']
  gamma[j] = best_parameters['gamma']    
  for i in range(length):
    classifier = SVC(C=best_parameters['C'],kernel=kernel,  gamma=best_parameters['gamma'])
    classifier.fit(X_train, y_train)
    plot_pca_svm(X_train, y_train,classifier)
    filename = 'finalized_model.sav'
    pickle.dump(classifier, open(filename, 'wb'))
#    importances = classifier.feature_importances_
#    print(importances)
#    for feature in zip(classifier.feature_importances_):
#        feature=feature
#        print(feature)
    y_pred = classifier.predict(X_test)
    acc_mat[i] = (accuracy_score(y_test, y_pred))
    f1[i]=f1_score(y_test, y_pred, average='weighted')
    std[i]=np.std(f1)
  acc_score[j] = acc_mat.mean()
  f1_avg[j] = f1.mean()
  std_avg[j] = std.mean()
  confMatrix = confusion_matrix(y_test, y_pred)
# print(confMatrix)
# print('Cross Validation Scores',cross_val_score(classifier, X, y, cv=5))
# print('Mean CV Score',np.mean(cross_val_score(classifier, X, y, cv=5)))
 # tp=confMatrix[0,0]
 # fp=confMatrix[0,1]
 # fn=confMatrix[1,0]
 # tn=confMatrix[1,1]
 # recall = (tp/(tp+fn)) #sensitivity tpr
 # print('recall or sesitivity:', recall)
 # fallout = (fp/(fp+tn)) #fpr
 # print('fallout:',fallout)
 # precision = (tp/(tp+fp))
 # print('precision:',precision)
 # specificity = 1 - fallout #tnr
 # print('specificity:',specificity)
 # f1 = ((2*tp)/(2*tp+fp+fn))
 # print('f1:',f1)
 # mcc = ((tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)
 # print('mcc:',mcc)
 # auc = ((recall+specificity)/2)
 # print('auc:',auc)
# print(metrics.classification_report(y_test, y_pred))
 target_names = ['class 0', 'class 1']
 report = metrics.classification_report(y_test, y_pred, target_names=target_names)
 ax = plt.gca()
 rfc_disp = plot_roc_curve(classifier, X_test, y_test, ax=ax, alpha=0.8)
 plt.show()
 return [acc_score, f1_avg,std_avg,C,kernel,gamma]

def result_lr( X_train, y_train ,X_test, y_test) :
 classifier = LogisticRegression()
 length = M
 cross_validation = CV
 acc_score = np.empty(length)
 C = np.empty(length)
 acc_mat = np.empty(length)
 f1 = np.empty(length)
 f1_avg = np.empty(length)
 std= np.empty(length)
 std_avg = np.empty(length)
 for j in range(length):
  grid_param = {
       'C': [0.01, 0.1, 1, 10, 100, 1000],  
       'penalty': ['l1','l2']
  }
  gd_sr = GridSearchCV(estimator=classifier,
                       param_grid=grid_param,
                       scoring='accuracy',
                       cv=cross_validation,
                       n_jobs=-1)
  gd_sr.fit(X_train, y_train)
#  print("Grid Search Best Score:", gd_sr.best_score_)
  warnings.filterwarnings('ignore')
  warnings.filterwarnings(action='ignore',category=DeprecationWarning)
  warnings.filterwarnings(action='ignore',category=FutureWarning)
  best_parameters = gd_sr.best_params_
  C[j] = best_parameters['C']
  penalty = best_parameters['penalty']   
  for i in range(length):
    classifier = LogisticRegression(C=best_parameters['C'],penalty=penalty)
    classifier.fit(X_train, y_train)
#    importances = classifier.feature_importances_
#    print(importances)
#        print(feature)
    y_pred = classifier.predict(X_test)
    acc_mat[i] = (accuracy_score(y_test, y_pred))
    f1[i]=f1_score(y_test, y_pred, average='weighted')
    std[i]=np.std(f1)
  acc_score[j] = acc_mat.mean()
  f1_avg[j] = f1.mean()
  std_avg[j] = std.mean()
 confMatrix = confusion_matrix(y_test, y_pred)
# print(confMatrix)
# print('Cross Validation Scores',cross_val_score(classifier, X, y, cv=5))
# print('Mean CV Score',np.mean(cross_val_score(classifier, X, y, cv=5)))
 # tp=confMatrix[0,0]
 # fp=confMatrix[0,1]
 # fn=confMatrix[1,0]
 # tn=confMatrix[1,1]
 # recall = (tp/(tp+fn)) #sensitivity tpr
 # print('recall or sesitivity:', recall)
 # fallout = (fp/(fp+tn)) #fpr
 # print('fallout:',fallout)
 # precision = (tp/(tp+fp))
 # print('precision:',precision)
 # specificity = 1 - fallout #tnr
 # print('specificity:',specificity)
 # f1 = ((2*tp)/(2*tp+fp+fn))
 # print('f1:',f1)
 # mcc = ((tp*tn-fp*fn)/((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**0.5)
 # print('mcc:',mcc)
 # auc = ((recall+specificity)/2)
 # print('auc:',auc)
# print(metrics.classification_report(y_test, y_pred))
 target_names = ['class 0', 'class 1']
 report = metrics.classification_report(y_test, y_pred, target_names=target_names)
 classification_report_csv(report)
 ax = plt.gca()
 rfc_disp = plot_roc_curve(classifier, X_test, y_test, ax=ax, alpha=0.8)
 plt.show()
 return [acc_score,f1_avg,std_avg, C,penalty]

rawdata = readFile('percentileDatasetCombined.csv')
data=rawdata
dataWithLabel=data
labels = data[0:,[22]] # For oxxygenation data[0:,[23]]  & for complication data[0:,[22]] 
data=data[0:,[1,2,3,4,5,6,7,8,14,15,16,17,18,19,20,21]]
data = np.array(data).astype(float)

X=data
y=labels
y=y.ravel()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=False)
imp_mean = SimpleImputer(missing_values=-1, strategy='mean')
imp_mean=imp_mean.fit(X_train)
X_train = imp_mean.transform(X_train)

imp_mean2 = SimpleImputer(missing_values=-1, strategy='mean')
imp_mean2=imp_mean2.fit(X_test)
X_test = imp_mean2.transform(X_test)

# sc = StandardScaler()
# X_train = sc.fit_transform(X_train)
# X_test = sc.transform(X_test)
# print("Number transactions X_train dataset: ", X_train.shape)
# print("Number transactions y_train dataset: ", y_train.shape)
# print("Number transactions X_test dataset: ", X_test.shape)
# print("Number transactions y_test dataset: ", y_test.shape)


sm = SMOTE(random_state=100)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train.ravel())

print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))



# X_train_res=X_train
# y_train_res=y_train

print('==========Combine Data RF=======')
[acc_score, f1_avg,std_avg, depth,n_estimator] = result_rf(X_train_res,y_train_res, X_test,y_test)
print("Accuracy Score:",acc_score.mean())
print("F1 Score:",f1_avg.mean())
print("Standard Deviation for F1 Score:",std_avg.mean())
print('Max_Depth:',depth.mean())
print('N_estimators:',n_estimator.mean())
print('\n')


print('==========Combine Data SVM=======\n')
[acc_score,f1_avg,std_avg, C,kernel,gamma] = result_svm(X_train_res,y_train_res, X_test,y_test)
print("Accuracy Score:",acc_score.mean())
print("F1 Score:",f1_avg.mean())
print("Standard Deviation for F1 Score:",std_avg.mean())
print('C:',C.mean())
print('Kernel:',kernel)
print('gamma:',gamma.mean())


# print('==========Combine Data LR=======')
# [acc_score, f1_avg, std_avg,C,penalty] = result_lr(X_train_res,y_train_res, X_test,y_test)
# print("Accuracy Score:",acc_score.mean())
# print("F1 Score:",f1_avg.mean())
# print("Standard Deviation for F1 Score:",std_avg.mean())
# print('C:',C.mean())
# print('penalty:',penalty)
# print('\n')

X_train_pre = X_train_res[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
X_test_pre = X_test[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
print('==========Intraoperative Data RF=======')
[acc_score,f1_avg,std_avg, depth,n_estimator] = result_rf(X_train_pre,y_train_res, X_test_pre,y_test)
print("Accuracy Score:",acc_score.mean())
print("F1 Score:",f1_avg.mean())
print("Standard Deviation for F1 Score:",std_avg.mean())
print('Max_Depth:',depth.mean())
print('N_estimators:',n_estimator.mean())
print('\n')

X_train_pre = X_train_res[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
X_test_pre = X_test[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
print('==========Intraoperative Data SVM=======\n')
[acc_score, f1_avg,std_avg, C,kernel,gamma] = result_svm(X_train_pre,y_train_res, X_test_pre,y_test)
print("Accuracy Score:",acc_score.mean())
print("F1 Score:",f1_avg.mean())
print("Standard Deviation for F1 Score:",std_avg.mean())
print('C:',C.mean())
print('Kernel:',kernel)
print('gamma:',gamma.mean())

X_train_pre = X_train_res[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
X_test_pre = X_test[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
print('==========Intraoperative Data LR=======')
[acc_score,f1_avg,std_avg, C,penalty] = result_lr(X_train_pre,y_train_res, X_test_pre,y_test)
print("Accuracy Score:",acc_score.mean())
print("F1 Score:",f1_avg.mean())
print("Standard Deviation for F1 Score:",std_avg.mean())
print('C:',C.mean())
print('penalty:',penalty)
print('\n')


# X_train_post = X_train_res[:,[12,13,14,15]]
# X_test_post = X_test[:,[12,13,14,15]]
# print('==========Preoperative Data RF=======')
# [acc_score, f1_avg,std_avg, depth,n_estimator] = result_rf(X_train_post,y_train_res, X_test_post,y_test)
# print("Accuracy Score:",acc_score.mean())
# print("F1 Score:",f1_avg.mean())
# print("Standard Deviation for F1 Score:",std_avg.mean())
# print('Max_Depth:',depth.mean())
# print('N_estimators:',n_estimator.mean())
# print('\n')

# X_train_post = X_train_res[:,[12,13,14,15]]
# X_test_post = X_test[:,[12,13,14,15]]
# print('==========Preoperative Data SVM=======')
# [acc_score, f1_avg,std_avg, C,kernel,gamma] = result_svm(X_train_post,y_train_res, X_test_post,y_test)
# print("Accuracy Score:",acc_score.mean())
# print("F1 Score:",f1_avg.mean())
# print("Standard Deviation for F1 Score:",std_avg.mean())
# print('C:',C.mean())
# print('Kernel:',kernel)
# print('gamma:',gamma.mean())


# X_train_pre = X_train_res[:,[12,13,14,15]]
# X_test_pre = X_test[:,[12,13,14,15]]
# print('==========Preoperative Data LR=======')
# [acc_score,f1_avg,std_avg, C,penalty] = result_lr(X_train_pre,y_train_res, X_test_pre,y_test)
# print("Accuracy Score:",acc_score.mean())
# print("F1 Score:",f1_avg.mean())
# print("Standard Deviation for F1 Score:",std_avg.mean())
# print('C:',C.mean())
# print('penalty:',penalty)
# print('\n')


# X_train_pre = X_train_res[:,[0,1,2,3,11]]
# X_test_pre = X_test[:,[0,1,2,3,11]]
# print('==========Intraoperative no percentile Data RF=======')
# [acc_score, f1_avg,std_avg, depth,n_estimator] = result_rf(X_train_pre,y_train_res, X_test_pre,y_test)
# print("Accuracy Score:",acc_score.mean())
# print("F1 Score:",f1_avg.mean())
# print("Standard Deviation for F1 Score:",std_avg.mean())
# print('Max_Depth:',depth.mean())
# print('N_estimators:',n_estimator.mean())
# print('\n')

# X_train_pre = X_train_res[:,[0,1,2,3,11]]
# X_test_pre = X_test[:,[0,1,2,3,11]]
# print('==========Intraoperative Data no percentile SVM=======\n')
# [acc_score, f1_avg,std_avg, C,kernel,gamma] = result_svm(X_train_pre,y_train_res, X_test_pre,y_test)
# print("Accuracy Score:",acc_score.mean())
# print("F1 Score:",f1_avg.mean())
# print("Standard Deviation for F1 Score:",std_avg.mean())
# print('C:',C.mean())
# print('Kernel:',kernel)
# print('gamma:',gamma.mean())

# X_train_pre = X_train_res[:,[0,1,2,3,11]]
# X_test_pre = X_test[:,[0,1,2,3,11]]
# print('==========Intraoperative Data no percentile LR=======')
# [acc_score, f1_avg,std_avg, C,penalty] = result_lr(X_train_pre,y_train_res, X_test_pre,y_test)
# print("Accuracy Score:",acc_score.mean())
# print("F1 Score:",f1_avg.mean())
# print("Standard Deviation for F1 Score:",std_avg.mean())
# print('C:',C.mean())
# print('penalty:',penalty)
# print('\n')

# X_train_pre = X_train_res[:,[4,5,6,7,8,9,10]]
# X_test_pre = X_test[:,[4,5,6,7,8,9,10]]
# print('==========Intraoperative  percentile Data RF=======')
# [acc_score, f1_avg,std_avg, depth,n_estimator] = result_rf(X_train_pre,y_train_res, X_test_pre,y_test)
# print("Accuracy Score:",acc_score.mean())
# print("F1 Score:",f1_avg.mean())
# print("Standard Deviation for F1 Score:",std_avg.mean())
# print('Max_Depth:',depth.mean())
# print('N_estimators:',n_estimator.mean())
# print('\n')

# X_train_pre = X_train_res[:,[4,5,6,7,8,9,10]]
# X_test_pre = X_test[:,[4,5,6,7,8,9,10]]
# print('==========Intraoperative Data percentile SVM=======\n')
# [acc_score, f1_avg,std_avg, C,kernel,gamma] = result_svm(X_train_pre,y_train_res, X_test_pre,y_test)
# print("Accuracy Score:",acc_score.mean())
# print("F1 Score:",f1_avg.mean())
# print("Standard Deviation for F1 Score:",std_avg.mean())
# print('C:',C.mean())
# print('Kernel:',kernel)
# print('gamma:',gamma.mean())

# X_train_pre = X_train_res[:,[4,5,6,7,8,9,10]]
# X_test_pre = X_test[:,[4,5,6,7,8,9,10]]
# print('==========Intraoperative Data percentile LR=======')
# [acc_score, f1_avg,std_avg, C,penalty] = result_lr(X_train_pre,y_train_res, X_test_pre,y_test)
# print("Accuracy Score:",acc_score.mean())
# print("F1 Score:",f1_avg.mean())
# print("Standard Deviation for F1 Score:",std_avg.mean())
# print('C:',C.mean())
# print('penalty:',penalty)
# print('\n')

# X_train_pre = X_train_res[:,[0,1,2,3,11,12,13,14,15]]
# X_test_pre = X_test[:,[0,1,2,3,11,12,13,14,15]]
# print('==========Combined without  percentile Data RF=======')
# [acc_score, f1_avg, std_avg,depth,n_estimator] = result_rf(X_train_pre,y_train_res, X_test_pre,y_test)
# print("Accuracy Score:",acc_score.mean())
# print("F1 Score:",f1_avg.mean())
# print("Standard Deviation for F1 Score:",std_avg.mean())
# print('Max_Depth:',depth.mean())
# print('N_estimators:',n_estimator.mean())
# print('\n')

# X_train_pre = X_train_res[:,[0,1,2,3,11,12,13,14,15]]
# X_test_pre = X_test[:,[0,1,2,3,11,12,13,14,15]]
# print('==========Combined without percentile SVM=======\n')
# [acc_score, f1_avg,std_avg, C,kernel,gamma] = result_svm(X_train_pre,y_train_res, X_test_pre,y_test)
# print("Accuracy Score:",acc_score.mean())
# print("F1 Score:",f1_avg.mean())
# print("Standard Deviation for F1 Score:",std_avg.mean())
# print('C:',C.mean())
# print('Kernel:',kernel)
# print('gamma:',gamma.mean())

# X_train_pre = X_train_res[:,[0,1,2,3,11,12,13,14,15]]
# X_test_pre = X_test[:,[0,1,2,3,11,12,13,14,15]]
# print('==========Combined without percentile LR=======')
# [acc_score, f1_avg,std_avg, C,penalty] = result_lr(X_train_pre,y_train_res, X_test_pre,y_test)
# print("Accuracy Score:",acc_score.mean())
# print("F1 Score:",f1_avg.mean())
# print("Standard Deviation for F1 Score:",std_avg.mean())
# print('C:',C.mean())
# print('penalty:',penalty)
# print('\n')
