import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df=pd.read_csv("D:/Data Science/drug200.csv")
x=df[['Age','Sex','BP','Cholesterol','Na_to_K']].values
y=df['Drug'].values

le_sex = preprocessing.LabelEncoder()
x[:, 1] = le_sex.fit_transform(x[:, 1])

le_bp = preprocessing.LabelEncoder()
x[:, 2] = le_bp.fit_transform(x[:, 2])

le_chol = preprocessing.LabelEncoder()
x[:, 3] = le_chol.fit_transform(x[:, 3])

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=3)

drug_tree=DecisionTreeClassifier(criterion="entropy",random_state=3)#criterion can be "gini" alsoi
drug_tree.fit(x_train,y_train)

y_pred=drug_tree.predict(x_test)

print(y_pred)

accuracy = metrics.accuracy_score(y_test, y_pred)
print("Accuracy of Decision tree is ",accuracy*100,"%")

plt.figure(figsize=(12, 8))
plot_tree(drug_tree, feature_names=['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K'], 
          class_names=drug_tree.classes_, filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()

