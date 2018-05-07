from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.externals.six import StringIO
import pydotplus
from sklearn import tree

# Create RandomForestClassifier object
skullsForest = RandomForestClassifier(n_estimators=10, criterion='entropy')

# Read data from the file
my_data = pd.read_csv('skulls-data.csv', delimiter=',')

# Separate the features in the file
featuresNames = list(my_data.columns.values)[3:7]

# Remove the column containing the target name since it doesn't contain numeric values.
# axis=1 means we are removing columns instead of rows.
X = my_data.drop(my_data.columns[[0, 1, 2]], axis=1).values

# Get target names from the data
targetNames = my_data['epoch'].unique().tolist()
y = my_data['epoch']

# Cross Validation of data using train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

skullsForest.fit(X_trainset, y_trainset)
predForest = skullsForest.predict(X_testset)

# Print the results
print(predForest)
print(y_testset)
print("RandomForests's Accuracy: ", metrics.accuracy_score(y_testset, predForest))

# Visualize the data
dot_data = StringIO()
filename = 'skullforests.png'

# Replace the argument for skullsForest below with the tree number to view that tree
tree.export_graphviz(skullsForest[9], out_file=dot_data,
                     feature_names=featuresNames,
                     class_names=targetNames,
                     filled=True, rounded=True,
                     special_characters=True,
                     leaves_parallel=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

# Write the output to the file
graph.write_png(filename)
print('File Created')
