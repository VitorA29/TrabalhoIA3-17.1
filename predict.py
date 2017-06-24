from sklearn import decomposition, metrics
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import RandomizedLasso
import random

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Util import *

def prediction():
    data = pack2Data()

    text_clf = decisionTree()

    text_clf = text_clf.fit(data.data, data.target)

    docs_test = data.data
    predicted = text_clf.predict(docs_test)
    print(np.mean(predicted == data.target))

    print(metrics.classification_report(data.target, predicted,
                                        target_names=categories))

    metrics.confusion_matrix(data.target, predicted)

#0.93796864349

def naiveBayes():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])

    return text_clf

def randomForest():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', RandomForestClassifier(n_estimators=10)),
                         ])

    return text_clf

def svc():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', LinearSVC()),
                         ])

    return text_clf

def decisionTree():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('reduce_dim', PCA()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', DecisionTreeClassifier()),
                         ])

    return text_clf

prediction()