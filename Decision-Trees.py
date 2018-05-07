import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
from sklearn import tree
import pydotplus
import matplotlib.image as mpimg

"""Info on dataset:-
epoch - The epoch the skull as assigned to, a factor with levels c4000BC c3300BC, c1850BC, c200BC, and cAD150, where 
the years are only given approximately.

mb - Maximal Breadth of the skull.
bh - Basiregmatic Heights of the skull.
bl - Basilveolar Length of the skull.
nh - Nasal Heights of the skull.
"""

# Read the data from the file
my_data = pd.read_csv('skulls-data.csv', delimiter=',')
print(my_data[0:5])

# Collect the feature names from the data
featuresNames = list(my_data.columns.values)[3:7]
print(featuresNames)

# Remove the column containing the target name since it doesn't contain numeric values.
# axis=1 means we are removing columns instead of rows.
X = my_data.drop(my_data.columns[[0, 1, 2]], axis=1).values
print(X[0:5])

# Collect the Target Names from the data
targetNames = my_data['epoch'].unique().tolist()
print(targetNames)

y = my_data['epoch']
print(y[0:5])

# Cross Validation of data using train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

# check shapes of datasets for train and test
print(X_trainset.shape)
print(y_trainset.shape)
print(X_testset.shape)
print(y_testset.shape)

# Create a Decision Tree Classifier object based on entropy
skullsTree = DecisionTreeClassifier(criterion='entropy')

# Training the data
skullsTree.fit(X_trainset, y_trainset)

# Prediction
predTree = skullsTree.predict(X_testset)

print(predTree[0:5])
print(y_testset[0:5])

print("DecisionTree's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

# Visualize Decision Tree
dot_data = StringIO()
filename = 'skulltree.png'
out = tree.export_graphviz(skullsTree, feature_names=featuresNames,
                           out_file=dot_data,
                           class_names=np.unique(y_trainset),
                           filled=True,
                           special_characters=True,
                           rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Write it to a file
graph.write_png(filename)

# Reading the Image from the file and displaying it
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img, interpolation='nearest')
plt.show()
print('Visualized')
