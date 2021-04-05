from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

#задание 1
col_names = ["sepal length", "sepal width", "petal length", "petal width", "class"]
feature_names = ["sepal length", "sepal width", "petal length", "petal width"]
iris_data = pd.read_csv("iris.data",names=col_names)
X = iris_data[feature_names]
kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
y_km = kmeans.fit_predict(X)
fig = plt.figure()
ax = fig.add_subplot()
ax.set_xlabel('sepal length')
ax.set_ylabel('sepal width')
x = np.array(iris_data['sepal length'])
y = np.array(iris_data['sepal width'])
plt.scatter(x,y,c=y_km)
#legend1 = ax.legend(*sctr.legend_elements(), loc="lower right", title="Classes")
#ax.add_artist(legend1)
plt.savefig("Iris_cluster.png")
#%%
#задание 2
customer_data = pd.read_csv("customers.csv")
feature_names = ["Age","Education","YearsEmployed","Income","CardDebt","OtherDebt",
                 "DebtIncomeRatio"]
X = customer_data[feature_names]
kmeans = KMeans(n_clusters=3)
kmeans = kmeans.fit(X)
y_km = kmeans.fit_predict(X)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')

x = np.array(customer_data['Education'])
y = np.array(customer_data['Age'])
z = np.array(customer_data['Income'])

sctr = ax.scatter(x,y,z, marker="s", c=y_km)
legend1 = ax.legend(*sctr.legend_elements(), loc="upper left", title="Classes")
ax.add_artist(legend1)
plt.savefig("Customer_cluster.png")