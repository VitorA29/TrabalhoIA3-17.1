from sklearn import decomposition, metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
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

    if(gridSearch):
        parameters = {'clf__min_samples_split': range(10, 500, 20), 'clf__max_depth': range(1, 20, 2)}

        text_clf = GridSearchCV(text_clf, parameters, n_jobs=1, verbose=1)
        text_clf.fit(data.data, data.target)
    else:
        text_clf = text_clf.fit(data.data, data.target)

    #nltk.download()

    testData = getTestData(type)
    docs_test = testData.data
    predicted = text_clf.predict(docs_test)

    #Escreve no arquivo txt.
    write2TxtFile(predicted, testData, data, type, classifier, showWrongPredict, showPredictions, gridSearch)

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

'''('slct', TruncatedSVD(n_components=2)),'''

def randomForest():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', RandomForestClassifier(n_estimators=10)),
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
                         ('clf', RFE(estimator=SVC(kernel="linear", C=1), n_features_to_select=3, step=1) ),
                         ])
    return text_clf

type = QUATERNARIO
gridSearch = True
showWrongPredictions = True
showPredictions = False
predict(DECISION_TREE, type, gridSearch, showWrongPredictions, showPredictions)
