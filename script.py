import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

# Setting up pandas so that it displays all columns instead of collapsing them
desired_width = 400
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 30)

# read csv
flags = pd.read_csv("flags.csv", header=0)

# inspect data
print(flags.columns)
print(flags.head())
print(flags[flags.Name == "Netherlands"])

# create labels and data to work with
labels = flags["Landmass"]
data = flags[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange"]]


# create train and test data/labels
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=1)

scores = []
for i in range(1, 21):
    # create classifier and fit training data
    tree = DecisionTreeClassifier(random_state=1, max_depth=i)
    tree.fit(train_data, train_labels)

    # append accuracy of model
    scores.append(tree.score(test_data, test_labels))

plt.plot(range(1, 21), scores)
plt.show()
plt.close('all')


# Add additional columns to data to get a better prediction
data = flags[["Red", "Green", "Blue", "Gold", "White", "Black", "Orange", "Circles",
              "Crosses", "Saltires", "Quarters", "Sunstars", "Crescent", "Triangle"]]

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=.2, random_state=1)

scores = []
for i in range(1, 21):
    tree = DecisionTreeClassifier(random_state=1, max_depth=i)
    tree.fit(train_data, train_labels)
    scores.append(tree.score(test_data, test_labels))

plt.plot(range(1, 21), scores)
plt.show()
# best depth is approx 4
plt.close('all')

