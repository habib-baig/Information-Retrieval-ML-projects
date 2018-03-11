from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn import tree
from sklearn import svm

print ("Naive Bayes Classifier results with 70% split:")
X_train, X_test, y_train, y_test = train_test_split(X_train_counts, reviewlabel, test_size=0.30, random_state=42)
clf = MultinomialNB().fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy score {:.4}%\n\n".format(accuracy_score(y_test, y_pred)*100))
print("Recall for food-relevant: {:.4}%\n\n".format(recall_score(y_test, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Recall for food-irrelevant: {:.4}%\n\n".format(recall_score(y_test, y_pred, pos_label='Food-irrelevant', average='binary')*100))
print("Precision for food-relevant: {:.4}%\n\n".format(precision_score(y_test, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Precision for food-irrelevant: {:.4}%\n\n".format(precision_score(y_test, y_pred, pos_label='Food-irrelevant', average='binary')*100))


print ("Naive Bayes Classifier results with 5-fold cross validation:")
#c_scores = cross_val_score(clf, X_train_counts, reviewlabel, cv=5)
y_pred = cross_val_predict(clf, X_train_counts, reviewlabel, cv=5)
print("Accuracy score {:.4}%\n\n".format(accuracy_score(reviewlabel, y_pred)*100))
print("Recall for food-relevant: {:.4}%\n\n".format(recall_score(reviewlabel, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Recall for food-irrelevant: {:.4}%\n\n".format(recall_score(reviewlabel, y_pred, pos_label='Food-irrelevant', average='binary')*100))
print("Precision for food-relevant: {:.4}%\n\n".format(precision_score(reviewlabel, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Precision for food-irrelevant: {:.4}%\n\n".format(precision_score(reviewlabel, y_pred, pos_label='Food-irrelevant', average='binary')*100))


print ("KNN Classifier results with 70% split:")
clf=KNeighborsClassifier(n_neighbors=5)
clf=clf.fit(X_train, y_train)
y_pred = nlf.predict(X_test)
print("Accuracy score {:.4}%\n\n".format(accuracy_score(y_test, y_pred)*100))
print("Recall for food-relevant: {:.4}%\n\n".format(recall_score(y_test, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Recall for food-irrelevant: {:.4}%\n\n".format(recall_score(y_test, y_pred, pos_label='Food-irrelevant', average='binary')*100))
print("Precision for food-relevant: {:.4}%\n\n".format(precision_score(y_test, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Precision for food-irrelevant: {:.4}%\n\n".format(precision_score(y_test, y_pred, pos_label='Food-irrelevant', average='binary')*100))

print ("KNN Classifier results with 5-fold cross validation:")
#c_scores = cross_val_score(clf, X_train_counts, reviewlabel, cv=5)
y_pred = cross_val_predict(clf, X_train_counts, reviewlabel, cv=5)
print("Accuracy score {:.4}%\n\n".format(accuracy_score(reviewlabel, y_pred)*100))
print("Recall for food-relevant: {:.4}%\n\n".format(recall_score(reviewlabel, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Recall for food-irrelevant: {:.4}%\n\n".format(recall_score(reviewlabel, y_pred, pos_label='Food-irrelevant', average='binary')*100))
print("Precision for food-relevant: {:.4}%\n\n".format(precision_score(reviewlabel, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Precision for food-irrelevant: {:.4}%\n\n".format(precision_score(reviewlabel, y_pred, pos_label='Food-irrelevant', average='binary')*100))

print ("Decision Tree Classifier results with 70% split:")
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)
y_pred = tlf.predict(X_test)
print("Accuracy score {:.4}%\n\n".format(accuracy_score(y_test, y_pred)*100))
print("Recall for food-relevant: {:.4}%\n\n".format(recall_score(y_test, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Recall for food-irrelevant: {:.4}%\n\n".format(recall_score(y_test, y_pred, pos_label='Food-irrelevant', average='binary')*100))
print("Precision for food-relevant: {:.4}%\n\n".format(precision_score(y_test, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Precision for food-irrelevant: {:.4}%\n\n".format(precision_score(y_test, y_pred, pos_label='Food-irrelevant', average='binary')*100))


print ("Decision Tree Classifier results with 5-fold cross validation:")
#c_scores = cross_val_score(clf, X_train_counts, reviewlabel, cv=5)
y_pred = cross_val_predict(clf, X_train_counts, reviewlabel, cv=5)
print("Accuracy score {:.4}%\n\n".format(accuracy_score(reviewlabel, y_pred)*100))
print("Recall for food-relevant: {:.4}%\n\n".format(recall_score(reviewlabel, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Recall for food-irrelevant: {:.4}%\n\n".format(recall_score(reviewlabel, y_pred, pos_label='Food-irrelevant', average='binary')*100))
print("Precision for food-relevant: {:.4}%\n\n".format(precision_score(reviewlabel, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Precision for food-irrelevant: {:.4}%\n\n".format(precision_score(reviewlabel, y_pred, pos_label='Food-irrelevant', average='binary')*100))

print ("SVM Classifier results with 70% split:")
clf = svm.SVC(kernel='linear', C=1)
clf=clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("Accuracy score {:.4}%\n\n".format(accuracy_score(y_test, y_pred)*100))
print("Recall for food-relevant: {:.4}%\n\n".format(recall_score(y_test, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Recall for food-irrelevant: {:.4}%\n\n".format(recall_score(y_test, y_pred, pos_label='Food-irrelevant', average='binary')*100))
print("Precision for food-relevant: {:.4}%\n\n".format(precision_score(y_test, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Precision for food-irrelevant: {:.4}%\n\n".format(precision_score(y_test, y_pred, pos_label='Food-irrelevant', average='binary')*100))


print ("SVM Classifier results with 5-fold cross validation:")
#c_scores = cross_val_score(clf, X_train_counts, reviewlabel, cv=5)
y_pred = cross_val_predict(clf, X_train_counts, reviewlabel, cv=5)
print("Accuracy score {:.4}%\n\n".format(accuracy_score(reviewlabel, y_pred)*100))
print("Recall for food-relevant: {:.4}%\n\n".format(recall_score(reviewlabel, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Recall for food-irrelevant: {:.4}%\n\n".format(recall_score(reviewlabel, y_pred, pos_label='Food-irrelevant', average='binary')*100))
print("Precision for food-relevant: {:.4}%\n\n".format(precision_score(reviewlabel, y_pred, pos_label='Food-relevant', average='binary')*100))
print("Precision for food-irrelevant: {:.4}%\n\n".format(precision_score(reviewlabel, y_pred, pos_label='Food-irrelevant', average='binary')*100))
