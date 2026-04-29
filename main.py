import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt 
import seaborn as sns

# Load the data
data = pd.read_csv("heart.csv")
print(data.info())
print(data.head())

#split
X = data.drop("target", axis=1)
y = data["target"]

# train and test 
X_train , X_test, y_train, y_test = train_test_split(X,y,test_size=0.2, random_state=42)

# Scaling 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Improving the performance
model = LogisticRegression(max_iter=1000, C=0.5)
model.fit(X_train, y_train)

#prediction
predictions = model.predict(X_test)

# confused matrix 
cm = confusion_matrix(y_test,predictions)
print(cm)

# precision and Recall
print("Accuracy: ", accuracy_score(y_test,predictions)) 
print("Precision: ", precision_score(y_test, predictions))
print("Recall: ", recall_score(y_test, predictions))

#graph
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Disease", "Disease"],
            yticklabels=["No Disease", "Disease"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()