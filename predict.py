from sklearn import decomposition, metrics
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import AdaBoostClassifier
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

def predict(classifier, type, gridSearch, showWrongPredict, showPredictions, rfeEnabled, pcaEnabled):
    data = getTrainData(type)

    text_clf = getClassifier(classifier, rfeEnabled, pcaEnabled)

    if(gridSearch and classifier == NAIVE_BAYES):
        parameters = {'clf__max_depth': range(1, 100)}

        parameters = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                      'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],}

        parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                      'tfidf__use_idf': (True, False),
                      'clf__alpha': (1e-2, 1e-3),
                      }

        text_clf = GridSearchCV(text_clf, parameters, n_jobs=1, verbose=1)
        text_clf = text_clf.fit(data.data, data.target)
    else:
        text_clf = text_clf.fit(data.data, data.target)

    #nltk.download()

    testData = getTestData(type)
    docs_test = testData.data
    predicted = text_clf.predict(docs_test)

    #print(text_clf.get_params()['slct'].explained_variance_ratio_.sum())

    #if(classifier == SVM):
        #print_top10(text_clf.get_params()['vect'], text_clf.get_params()['clf'], getCategory(type))

    mostInformative = []
    try:
        mostInformative = getMostInformative(10, text_clf)
    except:
        None

    #Escreve no arquivo txt.
    write2TxtFile(predicted, testData, data, type, classifier, showWrongPredict, showPredictions, gridSearch, rfeEnabled, pcaEnabled, mostInformative)
    #Escreve no arquivo .tex
    write2TexFile(predicted, testData, type, classifier, gridSearch, rfeEnabled, pcaEnabled)

def print_top10(vectorizer, clf, class_labels):
    """Prints features with the highest coefficient values, per class"""
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label,
              " ".join(feature_names[j] for j in top10)))

def getClassifier(classifier, rfeEnabled, pcaEnabled):
    if (classifier == DECISION_TREE):
        return decisionTree()
    elif (classifier == NAIVE_BAYES):
        return naiveBayes()
    elif (classifier == RANDOM_FOREST):
        return randomForest()
    elif(classifier == SVM):
        return svm(rfeEnabled, pcaEnabled)
    else:
        return ada()

def getMostInformative(nWords, text_clf):
    feature_names = text_clf.get_params()['vect'].get_feature_names()

    array = [x for (y, x) in sorted(zip(text_clf.get_params()['clf'].ranking_, feature_names))]

    return array[:nWords]

def naiveBayes():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', MultinomialNB()),
                         ])

    return text_clf

def randomForest():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', RandomForestClassifier(n_estimators=100)),
                         ])

    return text_clf

def decisionTree():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', DecisionTreeClassifier()),
                         ])
    return text_clf

def svm(rfeEnabled, pcaEnabled):
    if(rfeEnabled):
        if(pcaEnabled):
            text_clf = Pipeline([('vect', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('slct', TruncatedSVD(n_components=10)),
                                 ('clf', RFE(estimator=SVC(kernel="linear", C=1), n_features_to_select=3)),
                                 ])
        else:
            text_clf = Pipeline([('vect', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', RFE(estimator=SVC(kernel="linear", C=1), n_features_to_select=100)),
                                 ])
    else:
        if(pcaEnabled):
            text_clf = Pipeline([('vect', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('slct', TruncatedSVD(n_components=100)),
                                 ('clf', SVC(kernel="linear", C=1)),
                                 ])
        else:
            text_clf = Pipeline([('vect', CountVectorizer()),
                                 ('tfidf', TfidfTransformer()),
                                 ('clf', LinearSVC()),
                                 ])
    return text_clf

def ada():
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', AdaBoostClassifier(n_estimators=100)),
                         ])
    return text_clf

type = BINARIO
gridSearch = False
showWrongPredictions = True
showPredictions = False
rfeEnabled = False
pcaEnabled = True
predict(SVM, type, gridSearch, showWrongPredictions, showPredictions, rfeEnabled, pcaEnabled)
