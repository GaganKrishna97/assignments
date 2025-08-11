# ------------------------------------------------------------
# SECTION B - Q2: Decision Tree Classifier on Iris dataset
# ------------------------------------------------------------

# 1. Import necessary libraries
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 2. Load Iris dataset
iris = load_iris()
X = iris.data       # Features: Sepal length, Sepal width, Petal length, Petal width
y = iris.target     # Target: Species (0, 1, 2)

# 3. Split dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Initialize Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=42)

# 5. Train the model
clf.fit(X_train, y_train)

# 6. Predict on test data
y_pred = clf.predict(X_test)

# 7. Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Decision Tree Classifier Accuracy on test set: {accuracy:.2f}")


# 8. Visualize the trained decision tree
plt.figure(figsize=(15,10))
plot_tree(clf, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
plt.title("Decision Tree Trained on Iris Dataset")
plt.show()