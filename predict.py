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

def predict(classifier, type, gridSearch, showWrongPredict, showPredictions, cria_arq):
    data = getTrainData(type)

    if(classifier == DECISION_TREE):
        text_clf = decisionTree()
        opn = "a"
        output="######################DECISION_TREE######################\n"
    elif(classifier == NAIVE_BAYES):
        text_clf = naiveBayes()
        opn = "w"
        output="######################NAIVE_BAYES######################\n"
    elif(classifier == RANDOM_FOREST):
        text_clf = randomForest()
        opn = "a"
        output="######################RANDOM_FOREST######################\n"
    elif(classifier == SVM):
        text_clf = svm()
        opn = "a"
        output="######################SVM######################\n"
    else:
        text_clf = svc()
        opn = "a"
        output="######################SVC######################\n"

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

    #output += "Modo: " + getModoStr(type)
    print("Modo: ", getModoStr(type))
    output += "DataTrain length: " + str(len(data.data)) + "\n"
    output += "DataTest length: " + str(len(testData.data)) + "\n"

    docs_test = testData.data
    predicted = text_clf.predict(docs_test)
    output += "Accuracy: " + str(metrics.accuracy_score(testData.target, predicted)) + "\n"
    output += "\nReport:\n\n"
    output += str(metrics.classification_report(testData.target, predicted, target_names = getCategory(type))) + "\n"
    output += "\nConfusion Matrix:\n\n"
    output += str(metrics.confusion_matrix(testData.target, predicted))

    outputAux = ""
    if(showWrongPredict):
        outputAux += "\nWrong Predictions:\n"
        for i in getWrondPredictions(predicted, testData.target, docs_test):
            outputAux += i + "\n"

    if(showPredictions):
        array = []
        outputAux += "\nPredictions:\n"
        for i in range(0, len(predicted)):
            text = testData.data[i]
            classy = getCategory(type)[predicted[i]]
            textClass = TextClassification(text, classy)
            array.append(textClass)

        random.shuffle(array)
        j = 0
        for i in array:
            outputAux = j+1 + '-' + i.text + " -> " + i.classific + "\n"
            j += 1

    if (cria_arq == True):
        if(type==BINARIO):
            fo=open("BINARIO_RESULTADOS.txt",opn)
        elif(type==TERNARIO):
            fo=open("TERNARIO_RESULTADOS.txt",opn)
        else:
            fo=open("QUATERNARIO_RESULTADOS.txt",opn)
        fo.write(output + "\n")
        try:
            print_top10_file(text_clf.get_params()['vect'], text_clf.get_params()['clf'], getCategory(type), fo)
        except:
            None
        fo.write(outputAux)
        fo.close()
        return
    print(output)
    try:
        print_top10(text_clf.get_params()['vect'], text_clf.get_params()['clf'], getCategory(type))
    except:
        None
    print(outputAux)

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
        cria_arq = True
    elif (sys.argv[5]).upper() == "FALSE":
        cria_arq = False
    else:
        print("cria_arq invalido\n")
        break

    print("NAIVE BAYES")
    predict(NAIVE_BAYES, type, gridSearch, showWrongPredictions, showPredictions, cria_arq)
    print("RANDOM FOREST")
    predict(RANDOM_FOREST, type, gridSearch, showWrongPredictions, showPredictions, cria_arq)
    print("DECISION TREE")
    predict(DECISION_TREE, type, gridSearch, showWrongPredictions, showPredictions, cria_arq)
    print("SVM")
    predict(SVM, type, gridSearch, showWrongPredictions, showPredictions, cria_arq)
    break
