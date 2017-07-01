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

DECISION_TREE = 0
NAIVE_BAYES = 1
RANDOM_FOREST = 2
SVM = 3
SVC1 = 4

def predict(classifier, type):
    data = getTrainData(type)

    '''
        So chamar a função de cada classificador aqui.
        text_clf = naiveBayes()
        text_clf = randomForest()
        text_clf = svm()
    '''

    if(classifier == DECISION_TREE):
        text_clf = decisionTree()
    elif(classifier == NAIVE_BAYES):
        text_clf = naiveBayes()
    elif(classifier == RANDOM_FOREST):
        text_clf = randomForest()
    elif(classifier == SVM):
        text_clf = svm()
    else:
        text_clf = svc()

    '''
    parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
                  'tfidf__use_idf': (True, False),
                  'clf__alpha': (1e-2, 1e-3),
                  }
    gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)
    gs_clf = gs_clf.fit(data.data, data.target)
    #nltk.download()
    '''

    text_clf = text_clf.fit(data.data, data.target)

    testData = getTestData(type)

    print("DataTrain length: ", len(data.data))
    print("DataTest length: ", len(testData.data))

    docs_test = testData.data
    predicted = text_clf.predict(docs_test)
    print("Accuracy: ", metrics.accuracy_score(testData.target, predicted))
    print("\nReport:\n")
    print(metrics.classification_report(testData.target, predicted, target_names = getCategory(type)))
    print("\nConfusion Matrix:\n")
    print(metrics.confusion_matrix(testData.target, predicted))

    try:
        print_top10(text_clf.get_params()['vect'], text_clf.get_params()['clf'], getCategory(type))
    except:
        None

    if (len(sys.argv)>2 and sys.argv[1]=="false"):
         return

    array = []
    print("\nPredictions:")
    for i in range(0, len(predicted)):
        text = testData.data[i]
        classy = getCategory(type)[predicted[i]]
        textClass = TextClassification(text, classy)
        array.append(textClass)

    random.shuffle(array)
    j = 0
    for i in array:
        print(j+1,'-', i.text, " -> ", i.classific)
        j += 1

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

type = TERNARIO

print("RANDOM FOREST")
print(predict(RANDOM_FOREST, type))
print("NAIVE BAYES")
print(predict(NAIVE_BAYES, type))
print("DECISION TREE")
print(predict(DECISION_TREE, type))
print("SVM")
print(predict(SVM, type))
