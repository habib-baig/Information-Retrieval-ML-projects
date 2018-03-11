from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#

HyperParameters=[[(1,3),(1,2),(1,1)],[0.90,.85,0.75],[0.01,0.1,0.001,0.0001]]
for gram in HyperParameters[0]:
        for mx_df in HyperParameters[1]:
            for mn_df in HyperParameters[2]:
                print("For the parameters of: \nmax_df=",mx_df,"min_df=",mn_df,"\nngram_range=",gram)
                tfidf = TfidfVectorizer(stop_words=ENGLISH_STOP_WORDS,ngram_range=gram,max_df=mx_df,min_df=mn_df)

                X_tfidf = tfidf.fit_transform(reviewtext)
                X_train, X_test, y_train, y_test = train_test_split(X_tfidf, reviewlabel, test_size=0.30, random_state=42)
                clf = MultinomialNB().fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                print("Accuracy score {:.4}%\n\n".format(accuracy_score(y_test, y_pred)*100))
                print("Recall for food-relevant: {:.4}%\n\n".format(recall_score(y_test, y_pred, pos_label='Food-relevant', average='binary')*100))
                print("Recall for food-irrelevant: {:.4}%\n\n".format(recall_score(y_test, y_pred, pos_label='Food-irrelevant', average='binary')*100))
                print("Precision for food-relevant: {:.4}%\n\n".format(precision_score(y_test, y_pred, pos_label='Food-relevant', average='binary')*100))
                print("Precision for food-irrelevant: {:.4}%\n\n".format(precision_score(y_test, y_pred, pos_label='Food-irrelevant', average='binary')*100))


                print ("Naive Bayes Classifier results with 5-fold cross validation:")
                #c_scores = cross_val_score(clf, X_train_counts, reviewlabel, cv=5)
                y_pred = cross_val_predict(clf, X_tfidf, reviewlabel, cv=5)
                print("Accuracy score {:.4}%\n\n".format(accuracy_score(reviewlabel, y_pred)*100))
                print("Recall for food-relevant: {:.4}%\n\n".format(recall_score(reviewlabel, y_pred, pos_label='Food-relevant', average='binary')*100))
                print("Recall for food-irrelevant: {:.4}%\n\n".format(recall_score(reviewlabel, y_pred, pos_label='Food-irrelevant', average='binary')*100))
                print("Precision for food-relevant: {:.4}%\n\n".format(precision_score(reviewlabel, y_pred, pos_label='Food-relevant', average='binary')*100))
                print("Precision for food-irrelevant: {:.4}%\n\n".format(precision_score(reviewlabel, y_pred, pos_label='Food-irrelevant', average='binary')*100))
