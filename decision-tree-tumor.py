from sklearn.preprocessing import StandardScaler
st_x= StandardScaler()
x_train= st_x.fit_transform(x_train)
x_test= st_x.transform(x_test)


from sklearn.tree import DecisionTreeClassifier
tree_classification= DecisionTreeClassifier(random_State=1, criterion='entropy')
tree_classification.fit(x_train, y_train)

y_head= tree_classification.predict(x_test)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_head)
print("Accuracy of decision tree classification: {}".format(accuracy))

from sklearn.metrics import confusion_matrix
cm= confusion_matrix(y_test, y_head)

f,ax = plt.subplots(figsize=(5,5))
sns.heatmap(cm, annot=true, fnt= '.0f', linewidths=0.5, linecolor="red", ax=ax)
plt.xlabel("y_red")
plt.xlabel("y_test")
plt.show()