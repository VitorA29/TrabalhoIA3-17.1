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
    if not (len(sys.argv)>2 and (sys.argv[2]=="false" or sys.argv[2]=="FALSE")):
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
            outputAux += j+1 + '-' + i.text + " -> " + i.classific + "\n"
            j += 1

    if (len(sys.argv)>3 and (sys.argv[3]=="true" or sys.argv[3]=="TRUE")):
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

if sys.argv[1] == "TERNARIO":
    type = TERNARIO
elif sys.argv[1] == "BINARIO":
    type = BINARIO
else:
    type = QUATERNARIO

print("NAIVE BAYES")
predict(NAIVE_BAYES, type)
print("RANDOM FOREST")
predict(RANDOM_FOREST, type)
print("DECISION TREE")
predict(DECISION_TREE, type)
print("SVM")
predict(SVM, type)
