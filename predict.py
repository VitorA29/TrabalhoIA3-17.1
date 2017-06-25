from sklearn import decomposition, metrics
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import RandomizedLasso
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import SGDClassifier

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from Util import *

def predict():
    data = getTrainData()

    '''
        So chamar a função de cada classificador aqui.
        text_clf = naiveBayes()
        text_clf = randomForest()
        text_clf = svm()
    '''
    text_clf = decisionTree()


    text_clf = text_clf.fit(data.data, data.target)

    testData = getTestData()

    print("DataTrain length: ", len(data.data))
    print("DataTest length: ", len(testData.data))

    docs_test = testData.data
    predicted = text_clf.predict(docs_test)
    print("Accuracy: ", metrics.accuracy_score(testData.target, predicted))
    print("Report:\n")
    print(metrics.classification_report(testData.target, predicted, target_names = categories))
    print("Confusion Matrix:\n")
    print(metrics.confusion_matrix(testData.target, predicted))

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
                         ('tfidf', TfidfTransformer()),
                         ('clf', DecisionTreeClassifier()),
                         ])

    return text_clf

def svm():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, n_iter=5, random_state=42)),
                         ])
    return text_clf

predict()