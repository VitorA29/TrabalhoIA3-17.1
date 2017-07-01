import re, math, collections, itertools
import os

from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re, math, collections, itertools
import nltk, nltk.classify.util, nltk.metrics
from nltk.classify import NaiveBayesClassifier
from nltk.metrics import BigramAssocMeasures
from nltk.probability import FreqDist, ConditionalFreqDist

from nltk.classify import ClassifierI
from statistics import mode

'''
Classificador quaternário.
'''
categories4 = ['pos', 'neg', 'neu', 'irr']
'''
Classificador ternário.
'''
categories3 = ['pos', 'neg', 'neu']
'''
Classificador binário.
'''
categories2 = ['pos', 'neg']

QUATERNARIO_PATH = "./quaternario/"
TERNARIO_PATH = "./ternario/"
BINARIO_PATH = "./binario/"

QUATERNARIO = 0
TERNARIO = 1
BINARIO = 2

class Data(object):
    '''
        Essa classe encapsula as informações sobre os dados lidos.
        Por exemplo, li 5 arquivos. data quarda todos os textos e target seu valor(numérico)
        Os valores são positivo(0), negativo(1) e Neutro(2)
        Classe usada no treinamento.
    '''

    def __init__(self, text, category):
        self.data = text
        self.target = category

    def data(self):
        return self.data

    def target(self):
        return self.target

class TextClassification(object):

    def __init__(self, text, classific):
        self.text = text
        self.classific = classific

def text2Wordlist(text, removeStopwords = False):
    '''
    Dado um texto, transforma em um array de palavras.
    :param text: text a ser separado em palavras.
    :param removeStopwords: True caso queria que as stopwords sejam removidas, False(padrão) caso contrário.
    :return: array de palavras.
    '''

    #Remove qualquer token da sintaxe HTML.
    review_text = BeautifulSoup(text, "html.parser").get_text()

    #Remove qualquer caracter que não seja uma letra.
    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    #Converte todas as frases para lowercase e transforma num vetor de palavras.
    words = review_text.lower().split()

    #Remove as stopwords. Falso por padrão.
    if removeStopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]

    return(words)

def file2SentencesArray(fileName, folderName):
    '''
    Dado um arquivo .csv, joga todos elementos pra um array.
    :param fileName: Nome do arquivo sem a extenção.
    :return: retorna um array de comentário e se sua classificação.
    '''

    #header = 0 significa que a primeira linha do arquivo contem informações sobre o padrão do arquivo.
    #delimiter = "," significa que os dados estão separados por uma vírgula.
    #quoting = 3 remove aspas
    array = pd.read_csv(folderName + fileName, header=0,
                        delimiter=",")

    return array

def getTestData(type):
    dataArray = []
    targetArray = []

    path = getPath(type)

    for file in list_files(path + "test/"):
        array = file2SentencesArray(file, path + "test/")

        array1 = []
        for i in array["class"]:
            array1.append(getCategory(type).index(i[:3].lower()))

        dataArray.extend(cleanSentences(array["text"], False))
        targetArray.extend(array1)

    data = Data(dataArray, targetArray)

    return data

def getTrainData(type):
    dataArray = []
    targetArray = []

    path = getPath(type)

    for file in list_files(path + "train/"):
        array = file2SentencesArray(file, path + "train/")

        array1 = []
        for i in array["class"]:
            array1.append(getCategory(type).index(i[:3].lower()))

        dataArray.extend(cleanSentences(array["text"], True))
        targetArray.extend(array1)

    data = Data(dataArray, targetArray)

    return data

def  cleanSentences(sentencesArray, removeStopwords):
    '''
    Limpa uma frase de caracteres insedejados.
    :param sentencesArray: Array contendo frases.
    :return: retorna um array de frases sem certos caracteres.
    '''
    clean = []
    for i in range(0, len(sentencesArray)):
        clean.append(" ".join(text2Wordlist(sentencesArray[i], removeStopwords)))

    return clean

def getPath(type):
    if(type == BINARIO):
        return BINARIO_PATH
    elif(type == TERNARIO):
        return TERNARIO_PATH
    else:
        return QUATERNARIO_PATH

def getCategory(type):
    if(type == BINARIO):
        return categories2
    elif(type == TERNARIO):
        return categories3
    else:
        return categories4

def list_files(path):
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files

def print_top10(vectorizer, clf, class_labels):
    feature_names = vectorizer.get_feature_names()
    for i, class_label in enumerate(class_labels):
        top10 = np.argsort(clf.coef_[i])[-10:]
        print("%s: %s" % (class_label," ".join(feature_names[j] for j in top10)))

def show_most_informative_features(vectorizer, clf, n=20):
    feature_names = vectorizer.get_feature_names()
    coefs_with_fns = sorted(zip(clf.coef_[0], feature_names))
    top = zip(coefs_with_fns[:n], coefs_with_fns[:-(n + 1):-1])
    for (coef_1, fn_1), (coef_2, fn_2) in top:
        print("\t%.4f\t%-15s\t\t%.4f\t%-15s" % (coef_1, fn_1, coef_2, fn_2))