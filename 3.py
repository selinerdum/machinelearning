from sklearn.linear_model import LogisticRegression

x= [0, 1, 0, 1, 1, 0, 0, 0, 1, 1]
y= [0, 0, 0, 1, 1, 0, 1, 0, 1, 1]
# Modeli tanımlama
logreg = LogisticRegression(random_state=16)

# Veri ile eğitme
logreg.fit(X_train, y_train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()