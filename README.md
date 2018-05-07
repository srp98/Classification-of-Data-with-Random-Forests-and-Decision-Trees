# Classification-of-Data-with-Random-Forests-and-Decision-Trees
Random Forest Classification of data based on entropy, using RandomForestClassifier from Sci-Kit Learn module.

## Requirements
* Python 3.6 and above
* Modules needed-
  * sklearn
  * pydotplus
  * pandas
  * matplotlib
  * numpy

Install pydotplus before graphviz to avoid errors using pip.

## Data
Any labelled data can be used to test for random forests and decision trees. For now used a simple [Egyptian Skulls Dataset](https://www3.nd.edu/~busiforc/handouts/Data%20and%20Stories/regression/egyptian%20skull%20development/EgyptianSkulls.html)

## Running Code
- Execute the Decision trees and Random Forests python files respectively to check results, accuracy score and to visualize the Tree/Trees.
- Random Forests can be slow depending on the data you are working with as it creates multiple decision trees using bootstrapping, bigger the data slower the process.
