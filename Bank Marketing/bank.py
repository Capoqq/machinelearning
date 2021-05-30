# -*- coding: utf-8 -*-
"""
Created on Sat May 29 20:59:33 2021

@author: Janus
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import sklearn.neighbors import KNeighborsClassifier
import sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
url = 'bank-full.csv'
data = pd.read_csv(url)

#taratemiento de datos

data.job.replace(['blue-collar','management','technician','admin.','services','retired','self-employed','entrepreneur','unemployed','housemaid','student','unknown'],[0,1,2,3,4,5,6,7,8,9,10,11], inplace=True)
data.education.replace(['primary','secondary','tertiary','unknown'],[0,1,2,3], inplace=True)
data.marital.replace(['married','single','divorced'],[0,1,2], inplace=True)
data.default.replace(['no','yes'],[0,1], inplace=True)
data.drop(['housing','loan','contact','month','duration','campaign','pdays','previous','poutcome','y'], axis = 1,inplace =True)
data.age.mean()
data.age.replace(np.nan,41, inplace=True)
rangos = [0,8,15,18,25,40,60,100]
nombres = ['1','2','3','4','5','6','7']
data.age = pd.cut(data.age, rangos, labels = nombres)

data_train = data[:850]
data_test = data[850:]
x = np.array(data_train.drop(['balance'],1))
y = np.array(data_train.balance)
x_train,x_text,y_train,y_test = train_test_split(x,y,test_size=0.2)
x_test_out = np.array(data_test.drop(['balance'],1))
y_test_out = np.array(data_test.balance)

y_pred = logreg.predict(x_test_out)
print(accuracy_score(y_pred, y_test_out))
#matriz ed confusion
print(confusion_matrix(y_test_out, y_pred))
mostrar_resultados(y_test_out, y_pred)
probs = logreg.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)
fpr, trp, thresholds = roc_cuver(y_test_out, probs)
plot_roc_curve(fpr,tpr)
print(precision_score(y_test_out, y_pred))

# regresion logistica

logreg = LogisticRegression(solver = 'lbfgs', max_iter = 7600)
kfold = KFOLD(n?splits = 10)
cvscores = []
for train, test in kfold.split(x_train, y_train):
    logreg.fit(x_train[train], y_train[train])
    scrores = logreg.score(x_train[test], y_train[test])
    cvscores.append(scrores)
print(np.mean(cvscores))
print(accuracy_score(y_pred, y_test_out))
#matriz ed confusion
print(confusion_matrix(y_test_out, y_pred))
mostrar_resultados(y_test_out, y_pred)
probs = logreg.predict_proba(x_test_out)
probs = probs[:, 1]
auc = roc_auc_score(y_test_out, probs)
print('AUC: %.2f' % auc)
fpr, trp, thresholds = roc_cuver(y_test_out, probs)
plot_roc_curve(fpr,tpr)
print(precision_score(y_test_out, y_pred))

logreg.fit(x_train,y_train)
# Acuracy de entrenamiento
print(logreg.score(x_train, y_train))

#Acuracy de test
print(logreg.score(x_test, y_test))

# Maquina de soporte vectorial
svc = SVC(gamma = 'auto')
svc.fit(x_train,y_train)
print(svc.score(x_train, y_train))
print(svc.score(x_test, y_test))


#Vecinos mas cercanos clasificador

knn = KNeighborsClassifier(n_neighboars=3)
knn.fit(x_train,y_train)
print(knn.score(x_train, y_train))
print(knn.score(x_test, y_test))

def mostrar_resultados(r_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize = (6,6))
    sns.heatmap(conf_matrix,);
    plt.title("Confusion matrix")
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()
    print(classification_report(y_test,pred_y))


def plot_roc_curve(fpr, tpr):
    plt.plot(frp,tpr,color= 'orange',label = 'ROC')
    plt.plot([0,1], [0,1], color='darkblue',linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
    
    
model = DecisionTreeClassifier()
model, acc_validation, acc_test, y_pred =metricas_entrenamiento(model, x_train, x_test, y_train, y_test)
matriz_confusion, AUC, fpr, tpr  = matriz_confusion_auc(model, x_test, y_test, y_pred)
show_metrics('Decision Tree',AUC, acc_validation, acc_test, y_test, y_pred)
show_roc_curve_matrix(fpr, tpr, matriz_confusion)

#Bayesianos
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_pred  =  classifier.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
ac = accuracy_score(y_test,y_pred)