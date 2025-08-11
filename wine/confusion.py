# ------------------------------------------------------------
# Q4: Confusion Matrix for Iris Classification
# ------------------------------------------------------------
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# 1. Load data
iris = load_iris()
X = iris.data
y = iris.target

# 2. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Train model
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

# 4. Predictions
y_pred = clf.predict(X_test)


# 5. Confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)
disp.plot(cmap='Blues')


