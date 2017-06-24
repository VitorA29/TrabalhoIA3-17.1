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

def file2SentencesArray(fileName):
    '''
    Dado um arquivo .csv, joga todos elementos pra um array.
    :param fileName: Nome do arquivo sem a extenção.
    :return: retorna um array de comentário e se sua classificação.
    '''

    #header = 0 significa que a primeira linha do arquivo contem informações sobre o padrão do arquivo.
    #delimiter = "," significa que os dados estão separados por uma vírgula.
    #quoting = 3 remove aspas
    array = pd.read_csv(os.path.join(os.path.dirname(__file__), 'files', fileName + ".csv"), header=0,
                        delimiter=",", quoting=3)

    return array

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
