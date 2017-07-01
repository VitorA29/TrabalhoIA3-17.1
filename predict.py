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

def predict(classifier, type, gridSearch, showWrongPredict, showPredictions, fn):
    data = getTrainData(type)

    if(classifier == NAIVE_BAYES):
        text_clf = naiveBayes()
        imprimir("######################NAIVE_BAYES######################", fn)
    elif(classifier == DECISION_TREE):
        text_clf = decisionTree()
        imprimir("######################DECISION_TREE######################", fn)
    elif(classifier == RANDOM_FOREST):
        text_clf = randomForest()
        imprimir("######################RANDOM_FOREST######################", fn)
    elif(classifier == SVM):
        text_clf = svm()
        imprimir("######################SVM######################", fn)
    else:
        text_clf = svc()
        imprimir("######################SVC######################", fn)

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
        print("Best score: %0.3f" % text_clf.best_score_)
 
        '''
        best_parameters = text_clf.best_estimator_.get_params()
        for param_name in sorted(parameters.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
        '''
    else:
        text_clf = text_clf.fit(data.data, data.target)

    #nltk.download()

    testData = getTestData(type)

    print("Modo: ", getModoStr(type))
    imprimir("DataTrain length: " + str(len(data.data)), fn)
    imprimir("DataTest length: " + str(len(testData.data)), fn)

    docs_test = testData.data
    predicted = text_clf.predict(docs_test)
    imprimir("Accuracy: " + str(metrics.accuracy_score(testData.target, predicted)), fn)
    imprimir("\nReport:\n", fn)
    imprimir(str(metrics.classification_report(testData.target, predicted, target_names = getCategory(type))), fn)
    imprimir("\nConfusion Matrix:\n", fn)
    imprimir(str(metrics.confusion_matrix(testData.target, predicted)), fn)

    try:
        print_top10(text_clf.get_params()['vect'], text_clf.get_params()['clf'], getCategory(type), fn)
    except:
        None

    if(showWrongPredict):
        imprimir("\nWrong Predictions:", fn)
        for i in getWrondPredictions(predicted, testData.target, docs_test):
            imprimir(i, fn)

    if(showPredictions):
        array = []
        imprimir("\nPredictions:", fn)
        for i in range(0, len(predicted)):
            text = testData.data[i]
            classy = getCategory(type)[predicted[i]]
            textClass = TextClassification(text, classy)
            array.append(textClass)

        random.shuffle(array)
        j = 0
        for i in array:
            imprimir(j+1 + '-' + i.text + " -> " + i.classific, fn)
            j += 1

def getWrondPredictions(predictions, target, text):
    list = []

    for i in range(0, len(predictions)):
        if(predictions[i] != target[i]):
            list.append(text[i] + " | Correto: " + getCategory(type)[target[i]] + " Predição: " + getCategory(type)[predictions[i]])

    return list

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

while (len(sys.argv)==6):
    if (sys.argv[1]).upper() == "TERNARIO":
        type = TERNARIO
    elif (sys.argv[1]).upper() == "BINARIO":
        type = BINARIO
    elif (sys.argv[1]).upper() == "QUATERNARIO":
        type = QUATERNARIO
    else:
        print("type invalido\n")
        break

    if (sys.argv[2]).upper() == "TRUE":
        gridSearch = True
    elif (sys.argv[2]).upper() == "FALSE":
        gridSearch = False
    else:
        print("gridSearch invalido\n")
        break

    if (sys.argv[3]).upper() == "TRUE":
        showWrongPredictions = True
    elif (sys.argv[3]).upper() == "FALSE":
        showWrongPredictions = False
    else:
        print("showWrongPredictions invalido\n")
        break

    if (sys.argv[4]).upper() == "TRUE":
        showPredictions = True
    elif (sys.argv[4]).upper() == "FALSE":
        showPredictions = False
    else:
        print("showPredictions invalido\n")
        break

    if (sys.argv[5]).upper() == "TRUE":
        if(type==BINARIO):
            fn="BINARIO_RESULTADOS.txt"
        elif(type==TERNARIO):
            fn="TERNARIO_RESULTADOS.txt"
        else:
            fn="QUATERNARIO_RESULTADOS.txt"
    elif (sys.argv[5]).upper() == "FALSE":
        fn=""
    else:
        print("cria_arq invalido\n")
        break

    fo=open(fn, "w")
    fo.close()

    #print("NAIVE BAYES")
    predict(NAIVE_BAYES, type, gridSearch, showWrongPredictions, showPredictions, fn)
    #print("RANDOM FOREST")
    predict(RANDOM_FOREST, type, gridSearch, showWrongPredictions, showPredictions, fn)
    #print("DECISION TREE")
    predict(DECISION_TREE, type, gridSearch, showWrongPredictions, showPredictions, fn)
    #print("SVM")
    predict(SVM, type, gridSearch, showWrongPredictions, showPredictions, fn)
    break
