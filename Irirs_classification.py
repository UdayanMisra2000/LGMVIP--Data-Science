import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scikitplot as skplt

df=pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", names = ["sepal length", "sepal width", "petal length", "petal width", "class"])

#it will show 1st 5 values
df.head()
df.tail()

#no of rows and col
df.shape

#it will check if in the dataset any null value is present or not
df.isnull()

#it will show all the no of missing values in dataset
df.isnull().sum()

#show some statistical details
df.describe()

#show all cols
df.columns
#show all the no of unique elements in the object
df.nunique()
#it will return maximum value
df.max()
#it will return minimum value
df.min()

#visualization

#for visualization, boxplot() is using here.it has many parameters.Its showing here min , max , medium, 1st quartile and 3rd quartile
sns.boxplot(x="class", y="petal length", data=df)
plt.show()
sns.boxplot(x="class", y="sepal width", data=df)
plt.show()
sns.boxplot(x="class", y="sepal length", data=df)
plt.show()
sns.boxplot(x="class", y="petal width", data=df)
plt.show()
sns.boxplot( y="sepal width", data=df)
plt.show()
#this is the box plot for each column individually
sns.boxplot( y="sepal length", data=df)
plt.show()
sns.boxplot( y="petal width", data=df)
plt.show()
sns.boxplot( y="petal length", data=df)
plt.show()
#showing pair wise relationships in dataset using pairplot
sns.pairplot(df,hue="class")
#data preprocessing before modeling
#heatmap is used to show 2d data in graphical format.eact data value has a special color and represents in a matrix form.
#high intensity data will show high intensity color.
#annotation is true
#color map is used for coloring..we may change that.
plt.figure(figsize=(10,7))
sns.heatmap(df.corr(),annot=True,cmap="seismic")
plt.show()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
#fit_transform is used to label encoder and return encoded values
df['class']=le.fit_transform(df['class'])
df.head()
x=df.drop(columns=['class']) #dropping the column class and storing the values in x, without class
y=df['class'] #in y saving the class
x[:5] #it will return till index 5
y[:5]

#storing
lr=LogisticRegression()
knn=KNeighborsClassifier()
svm=SVC()
nb=GaussianNB()
dt=DecisionTreeClassifier()
rf=RandomForestClassifier()


print('Hi')
