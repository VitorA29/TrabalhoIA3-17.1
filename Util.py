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

categories = ['pos', 'neg', 'neu', 'irr']

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
    array = pd.read_csv(os.path.join(os.path.dirname(__file__), folderName, fileName), header=0,
                        delimiter=",")

    return array

def getTestData():
    dataArray = []
    targetArray = []

    for file in list_files("./test/"):
        array = file2SentencesArray(file, 'test')

        array1 = []
        for i in array["class"]:
            array1.append(categories.index(i[:3].lower()))

        dataArray.extend(cleanSentences(array["text"]))
        targetArray.extend(array1)

    data = Data(dataArray, targetArray)

    return data

def getTrainData():
    dataArray = []
    targetArray = []

    for file in list_files("./train/"):
        array = file2SentencesArray(file, 'train')

        array1 = []
        for i in array["class"]:
            array1.append(categories.index(i[:3].lower()))

        dataArray.extend(cleanSentences(array["text"]))
        targetArray.extend(array1)



    data = Data(dataArray, targetArray)

    return data

def  cleanSentences(sentencesArray):
    '''
    Limpa uma frase de caracteres insedejados.
    :param sentencesArray: Array contendo frases.
    :return: retorna um array de frases sem certos caracteres.
    '''
    clean = []
    for i in range(0, len(sentencesArray)):
        clean.append(" ".join(text2Wordlist(sentencesArray[i], True)))

    return clean

def list_files(path):
    files = []
    for name in os.listdir(path):
        if os.path.isfile(os.path.join(path, name)):
            files.append(name)
    return files