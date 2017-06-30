import sys

from sklearn import decomposition, metrics
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import RandomizedLasso
import random
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report

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

def predict(classifier):
    data = getTrainData()

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

    #nltk.download()

    text_clf = text_clf.fit(data.data, data.target)

    testData = getTestData()

    print("DataTrain length: ", len(data.data))
    print("DataTest length: ", len(testData.data))

    docs_test = testData.data
    predicted = text_clf.predict(docs_test)
    print("Accuracy: ", metrics.accuracy_score(testData.target, predicted))
    print("\nReport:\n")
    print(metrics.classification_report(testData.target, predicted, target_names = categories4))
    print("\nConfusion Matrix:\n")
    print(metrics.confusion_matrix(testData.target, predicted))

    try:
        print("\nTop 10 most informative words:\n")
        print_top10(text_clf.get_params()['vect'], text_clf.get_params()['clf'], categories4)
    except:
        print('It was not possible to find most informative words.')

    if (len(sys.argv)>2 and sys.argv[1]=="false"):
         return

    array = []
    print("\nPredictions:")
    for i in range(0, len(predicted)):
        text = testData.data[i]
        classy = categories4[predicted[i]]
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

if __name__ == "__main__":
    print(predict(NAIVE_BAYES))
