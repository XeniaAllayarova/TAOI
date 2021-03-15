import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np

accuracy_list = []
precision_list = []
recall_list = []
F_list = []
def secondTaskTree(testSize):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize)
    second_clf = [DecisionTreeClassifier(criterion="entropy"),DecisionTreeClassifier()]
    second_clf[0] = second_clf[0].fit(X_train,y_train)
    second_clf[1] = second_clf[1].fit(X_train,y_train)
    y_pred = [second_clf[0].predict(X_test),second_clf[1].predict(X_test)]

    accuracy_list.append([metrics.accuracy_score(y_test, y_pred[0]),
                          metrics.accuracy_score(y_test, y_pred[1])])
    precision_list.append([metrics.precision_score(y_test, y_pred[0],average='weighted',zero_division=0),
                           metrics.precision_score(y_test, y_pred[1],average='weighted',zero_division=0)])
    recall_list.append([metrics.recall_score(y_test, y_pred[0],average='weighted',zero_division=0),
                        metrics.recall_score(y_test, y_pred[1],average='weighted',zero_division=0)])
    F_list.append([metrics.f1_score(y_test, y_pred[0],average='weighted',zero_division=0),
                   metrics.f1_score(y_test, y_pred[1],average='weighted',zero_division=0)])
    plot_tree(second_clf[0])
    fig.savefig("second_inf_gain"+str(testSize*100)+".png")
    plot_tree(second_clf[1])
    fig.savefig("second_gini"+str(testSize*100)+".png")
    
data = pd.read_csv('grades.csv')
feature_cols = ['TEACHER_RIGHT','TEACHER_CHK','TEACHER_QUEST','TEACHER_CORR',
                'PUPIL_CORR','PUPIL_STRIP']

X = data[feature_cols]
y = data['GRADE']

#1. Построение дерева решений для классификации контрольных работ
first_clf = DecisionTreeClassifier()
first_clf = first_clf.fit(X,y)
fig = plt.figure(figsize=(50,50))
plot_tree(first_clf)
fig.savefig("first_decistion_tree.png")

#2. Разбиение данных на обучающую и тестовую выборки, а затем построение и 
# визуализация деревьев для различных атрибутов разбиения

secondTaskTree(0.4)
secondTaskTree(0.3)
secondTaskTree(0.2)
secondTaskTree(0.1)

#3. Вычисление показателей качества классификации
index = np.arange(4)
dataAccuracy = {'Accuracy inf gain': [accuracy_list[0][0],accuracy_list[1][0],
                                      accuracy_list[2][0],accuracy_list[3][0]],
                'Accuracy gini': [accuracy_list[0][1],accuracy_list[1][1],
                                  accuracy_list[2][1],accuracy_list[3][1]]}
df = pd.DataFrame(dataAccuracy)
df.plot(kind='bar')
plt.xticks(index,['60-40','70-30','80-20','90-10'])
plt.savefig("Accuracy.png")

dataPrecition = {'Precition inf gain': [precision_list[0][0],precision_list[1][0],
                                      precision_list[2][0],precision_list[3][0]],
                'Precition gini': [precision_list[0][1],precision_list[1][1],
                                  precision_list[2][1],precision_list[3][1]]}
df = pd.DataFrame(dataPrecition)
df.plot(kind='bar')
plt.xticks(index,['60-40','70-30','80-20','90-10'])
plt.savefig("Precition.png")

dataRecall = {'Recall inf gain': [recall_list[0][0],recall_list[1][0],
                                  recall_list[2][0],recall_list[3][0]],
              'Recall gini': [recall_list[0][1],recall_list[1][1],
                              recall_list[2][1],recall_list[3][1]]}
df = pd.DataFrame(dataRecall)
df.plot(kind='bar')
plt.xticks(index,['60-40','70-30','80-20','90-10'])
plt.savefig("Recall.png")

dataF = {'F-metric inf gain': [F_list[0][0],F_list[1][0],
                               F_list[2][0],F_list[3][0]],
         'F-metric gini': [F_list[0][1],F_list[1][1],
                           F_list[2][1],F_list[3][1]]}
df = pd.DataFrame(dataF)
df.plot(kind='bar')
plt.xticks(index,['60-40','70-30','80-20','90-10'])
plt.savefig("F-metric.png")