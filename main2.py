from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading Dataset
iris = datasets.load_iris()

# Printing descriptions & features
# print(iris.DESCR)
features = iris.data
labels = iris.target
#print(features[0],labels[0])

#Training the classifier
clf = KNeighborsClassifier()
clf.fit(features,labels)
pred = clf.predict([[1,1,1,1],[5,6,4,3],[4,3,7,5]])
print(pred)
