import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics, preprocessing
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
import matplotlib.pyplot as plt
import numpy as np

print("Задание 1\n")
col_names = ['age', 'workclass','fnlwgt','education','education-num','marital-status',
             'occupation','relationship','race','sex','capital-gain', 'capital-loss',
             'hours-per-week','native-country','year-salary']
data = pd.read_csv('adult.data',names=col_names)
feature_cols = ['age', 'workclass','fnlwgt','education','education-num','marital-status',
             'occupation','relationship','race','sex','capital-gain', 'capital-loss',
             'hours-per-week','native-country']

for col in ['workclass','education','marital-status','occupation','relationship',
            'race','sex','native-country']:
    le = preprocessing.LabelEncoder()
    le.fit(data[col])
    data[col]=le.transform(data[col])


X = data[feature_cols]
y = data['year-salary']
# Задание 1
clf = DecisionTreeClassifier()
clf = clf.fit(X,y)
# fig = plt.figure(figsize=(100,100))
# plot_tree(clf)
# fig.savefig("year_salary_tree.png")

test = pd.read_csv('adult.test',names=col_names)

for col in ['workclass','education','marital-status','occupation','relationship',
            'race','sex','native-country']:
    le = preprocessing.LabelEncoder()
    le.fit(test[col])
    test[col]=le.transform(test[col])
    
X_test = test[feature_cols]
Y_test = test['year-salary']
Y_pred = clf.predict(X_test)

accuracy = round(metrics.accuracy_score(Y_test, Y_pred),4)
precision = round(metrics.precision_score(Y_test, Y_pred,average='weighted',zero_division=0),4)
recall = round(metrics.recall_score(Y_test, Y_pred,average='weighted',zero_division=0),4)
F_metric = round(metrics.f1_score(Y_test, Y_pred,average='weighted',zero_division=0),4)
print("accuracy: "+str(accuracy*100)+"%\nprecision: "+str(precision*100)+"%\nrecall: "
      +str(recall*100)+"%\nF metric: "+str(F_metric*100)+"%")

#Задание 2
print("\nЗадание 2\n")
accuracy_list = []
precision_list = []
recall_list = []
F_list = []
for i in range (50,101,10):
    rclf = RandomForestClassifier(n_estimators=i)
    rclf = rclf.fit(X,y)
    Y_pred_RF = rclf.predict(X_test)
    aclf = AdaBoostClassifier(n_estimators=i)
    aclf = aclf.fit(X,y)
    Y_pred_B = aclf.predict(X_test)
    accuracy_list.append([metrics.accuracy_score(Y_test, Y_pred_RF),
                          metrics.accuracy_score(Y_test, Y_pred_B)])
    precision_list.append([metrics.precision_score(Y_test, Y_pred_RF, 
                                                   average='weighted',zero_division=0),
                           metrics.precision_score(Y_test, Y_pred_B, 
                                                   average='weighted',zero_division=0)])
    recall_list.append([metrics.recall_score(Y_test, Y_pred_RF,average='weighted',
                                            zero_division=0),
                        metrics.recall_score(Y_test, Y_pred_B,average='weighted',
                                            zero_division=0)])
    F_list.append([metrics.f1_score(Y_test, Y_pred_RF,average='weighted',zero_division=0),
                   metrics.f1_score(Y_test, Y_pred_B,average='weighted',zero_division=0)])

index = np.arange(6)
list_len = 6
dataAccuracy = {'Random Forest': [accuracy_list[i][0] for i in range(0,list_len)],
                'Boosting': [accuracy_list[i][1] for i in range(0,list_len)]}
df = pd.DataFrame(dataAccuracy)
df.plot(kind='bar',title='Accuracy')
plt.xticks(index,[str(x) for x in range(50,101,10)])
plt.ylim([0.84,0.87])
plt.savefig("Accuracy-2.png")

dataPrecision = {'Random Forest': [precision_list[i][0] for i in range(0,list_len)],
                 'Boosting': [precision_list[i][1] for i in range(0,list_len)]}
df = pd.DataFrame(dataPrecision)
df.plot(kind='bar',title='Precision')
plt.xticks(index,[str(x) for x in range(50,101,10)])
plt.ylim([0.84,0.87])
plt.savefig("Precision-2.png")

dataRecall = {'Random Forest': [recall_list[i][0] for i in range(0,list_len)],
              'Boosting': [recall_list[i][1] for i in range(0,list_len)]}
df = pd.DataFrame(dataRecall)
df.plot(kind='bar',title='Recall')
plt.xticks(index,[str(x) for x in range(50,101,10)])
plt.ylim([0.84,0.87])
plt.savefig("Recall-2.png")

dataF= {'Random Forest': [F_list[i][0] for i in range(0,list_len)],
        'Boosting': [F_list[i][1] for i in range(0,list_len)]}
df = pd.DataFrame(dataF)
df.plot(kind='bar',title='F metric')
plt.xticks(index,[str(x) for x in range(50,101,10)])
plt.ylim([0.84,0.87])
plt.savefig("F metric-2.png")
