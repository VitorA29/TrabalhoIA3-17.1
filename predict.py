import sys

from sklearn import decomposition, metrics
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import RFE
from sklearn.feature_selection import RFECV
from sklearn.linear_model import RandomizedLasso
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold

from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from Util import *

def predict(classifier, type, gridSearch, showWrongPredict, showPredictions):
    data = getTrainData(type)

    text_clf = getClassifier(classifier)

    if(gridSearch and classifier != SVM):
        print("------> A T E N Ç Ã O <------ GridSearch so funciona com SVM por enquanto! Executando sem GridSearch...")

    if(gridSearch and classifier == SVM):
        parameters = {
            'vect__max_df': (0.5, 0.75, 1.0),
            # 'vect__max_features': (None, 5000, 10000, 50000),
            'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
            # 'tfidf__use_idf': (True, False),
            # 'tfidf__norm': ('l1', 'l2'),
            'clf__alpha': (0.00001, 0.000001),
            'clf__penalty': ('l2', 'elasticnet'),
            # 'clf__n_iter': (10, 50, 80),
        }

        text_clf = GridSearchCV(text_clf, parameters, n_jobs=1, verbose=1)
        text_clf.fit(data.data, data.target)
    else:
        text_clf = text_clf.fit(data.data, data.target)

    #nltk.download()

    testData = getTestData(type)
    docs_test = testData.data
    predicted = text_clf.predict(docs_test)

    #Escreve no arquivo txt.
    wirte2TxtFile(predicted, testData, data, type, classifier, 'teste', showWrongPredict, showPredictions)

def getClassifier(classifier):
    if (classifier == DECISION_TREE):
        return decisionTree()
    elif (classifier == NAIVE_BAYES):
        return naiveBayes()
    elif (classifier == RANDOM_FOREST):
        return randomForest()
    else:
        return svm()

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
                         ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                               alpha=1e-3, n_iter=5, random_state=42))
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

type = QUATERNARIO
gridSearch = False
showWrongPredictions = True
showPredictions = True
predict(NAIVE_BAYES, type, gridSearch, showWrongPredictions, showPredictions)
