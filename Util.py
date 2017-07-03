import random
import re, math, collections, itertools
import os

from bs4 import BeautifulSoup
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import re, math, collections, itertools
from sklearn import decomposition, metrics
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

QUATERNARIO_PATH = "./bases/quaternario/"
TERNARIO_PATH = "./bases/ternario/"
BINARIO_PATH = "./bases/binario/"

BINARIO = 0
TERNARIO = 1
QUATERNARIO = 2


NAIVE_BAYES = 0
DECISION_TREE = 1
RANDOM_FOREST = 2
SVM = 3
ADA = 4

def getClassificadoresQTD():
    return 5

class Data(object):
    '''
        Essa classe encapsula as informações sobre os dados lidos.
        Por exemplo, li 5 arquivos. data quarda todos os textos e target seu valor(numérico)
        Os valores são positivo(0), negativo(1) e Neutro(2)
        Classe usada no treinamento.
    '''

    def __init__(self, text, category, usedFiles):
        self.data = text
        self.target = category
        self.usedFiles = usedFiles

    def data(self):
        return self.data

    def target(self):
        return self.target

    def usedFiles(self):
        return self.usedFiles

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

    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

    #Remove qualquer token da sintaxe HTML.
    review_text = BeautifulSoup(text, "html.parser").get_text()

    #Remove qualquer caracter que não seja uma letra.
    review_text = re.sub("[^a-zA-Z]"," ", review_text)

    #Converte todas as frases para lowercase e transforma num vetor de palavras.
    words = review_text.lower().split()

    #Remove as stopwords. Falso por padrão.
    if removeStopwords:
        stops = set(stopwords.words("english"))
        stops.add("http")
        stops.add("co")
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

    data = Data(dataArray, targetArray, list_files(path + "test/"))

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

    data = Data(dataArray, targetArray, list_files(path + "train/"))

    return data

def cleanSentences(sentencesArray, removeStopwords):
    '''
    Limpa uma frase de caracteres insedejados.
    :param sentencesArray: Array contendo frases.
    :return: retorna um array de frases sem certos caracteres.
    '''
    clean = []
    for i in range(0, len(sentencesArray)):
        clean.append(" ".join(text2Wordlist(sentencesArray[i], removeStopwords)))

    return clean

def getModoStr(type):
    if(type == BINARIO):
        return "Binário"
    elif(type == TERNARIO):
        return "Ternário"
    else:
        return "Quaternário"

def getModoStrDir(type):
    if(type == BINARIO):
        return "BINARIO"
    elif(type == TERNARIO):
        return "TERNARIO"
    else:
        return "QUATERNARIO"

def getClassifierName(classifier):
    if(classifier == DECISION_TREE):
        return 'Decision Tree'
    elif(classifier == NAIVE_BAYES):
        return 'Naive Bayes'
    elif(classifier == RANDOM_FOREST):
        return 'Random Forest'
    elif(classifier == SVM):
        return 'SVM'
    else:
        return 'ADA Boost'

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

def getWrongPredictions(predictions, target, text):
    list = []

    for i in range(0, len(predictions)):
        if(predictions[i] != target[i]):
            list.append("[Correct: " + getCategory(type)[target[i]] + " ; Predicted: " + getCategory(type)[predictions[i]] + "] " + text[i])

    return list

def write2TxtFile(predicted, testData, trainData, type, classifier, showWrongPredict, showPredictions, gridSearch, rfeEnabled, pcaEnabled, mostInformative):

    fname = getModoStrDir(type) + "/" + getClassifierName(classifier).upper()
    if gridSearch==True:
        fname += "_GRIDSEARCH"
    if rfeEnabled==True:
        fname += "_RFE"
    if pcaEnabled==True:
        fname += "_PCA"

    if not os.path.exists("execucao/" + getModoStrDir(type)):
        os.makedirs("execucao/" + getModoStrDir(type))
 
    text_file = open("execucao/" +  fname + ".txt", "w")
    print("Iteração atual: " + fname)

    text_file.write("Classifier: %s\n" % getClassifierName(classifier))
    text_file.write("Mode: %s\n" % getModoStr(type))
    text_file.write("GridSearch: %s\n" % (str(gridSearch)))
    text_file.write("RFE: %s\n" % (str(rfeEnabled)))
    text_file.write("PCA: %s\n" % (str(pcaEnabled)))
    text_file.write("Used files for training: ")
    for i in trainData.usedFiles:
        text_file.write("%s;" % i)

    text_file.write("\n")

    text_file.write("Used files for testing: ")
    for i in testData.usedFiles:
        text_file.write("%s;" % i)

    text_file.write("\n")

    text_file.write("DataTrain length: %s\n" % len(trainData.data))
    text_file.write("DataTest length: %s\n" % len(testData.data))

    text_file.write("\n-----PREDCTION INFO-----\n")

    text_file.write("Accuracy: %s\n" % metrics.accuracy_score(testData.target, predicted))
    text_file.write("\nReport:\n")
    text_file.write(metrics.classification_report(testData.target, predicted, target_names=getCategory(type)))
    text_file.write("\nConfusion Matrix:\n")
    text_file.write(str(metrics.confusion_matrix(testData.target, predicted)))

    if(rfeEnabled):
        j = 0
        text_file.write("\n\nMost Informative Words(RFE)\n")
        for i in mostInformative:
            text_file.write("%s - %s\n" % ((j + 1), i))
            j += 1

    if (showWrongPredict):
        list = getWrongPredictions(predicted, testData.target, testData.data)
        text_file.write("\n\n"+ str(len(list)) + " Wrong Predictions:\n")
        j = 0
        for i in list:
            text_file.write(str(j) + " - " + i + '\n')
            j += 1

    if (showPredictions):
        array = []
        text_file.write("\nPredictions:\n")
        for i in range(0, len(predicted)):
            text = testData.data[i]
            classy = getCategory(type)[predicted[i]]
            textClass = TextClassification(text, classy)
            array.append(textClass)

        random.shuffle(array)
        j = 0
        for i in array:
            text_file.write("%s - [%s]  %s\n" %((j + 1), i.classific, i.text))
            j += 1

    text_file.close()

def write2TexFile(predicted, testData, type, classifier, gridSearch, rfeEnabled, pcaEnabled):

    fname =getModoStrDir(type) + "/" + getClassifierName(classifier).upper()
    if gridSearch==True:
        fname += "_GRIDSEARCH"
    if rfeEnabled==True:
        fname += "_RFE"
    if pcaEnabled==True:
        fname += "_PCA"

    if not os.path.exists("execucao/tex/" + getModoStrDir(type)):
        os.makedirs("execucao/tex/" + getModoStrDir(type))

    print("Tex: " + fname)
    text_file = open("execucao/tex/" + fname +".tex", "w")

    comment = getClassifierName(classifier).lower() + "_" + getModoStrDir(type).lower()
    text_file.write("%" + comment + "\n")
    text_file.write("\\begin{table}[h!]\n")
    text_file.write("\\centering\n")
    text_file.write("\\begin{minipage}[b]{0.45\linewidth}\n")
    text_file.write("\\caption{Matriz de Confusão " + getModoStr(type) + ": \\textit{" + getClassifierName(classifier))
    if gridSearch:
        text_file.write(" com Grid Search")
    if rfeEnabled:
        text_file.write(" com RFE")
    if pcaEnabled:
        text_file.write(" com PCA")
    text_file.write("}}\n")
    text_file.write("\\label{tab:mcb-nb}\n")
    text_file.write("\\begin{tabular}{|l|l|l")
    for i in range(type):
        text_file.write("|l")
    text_file.write("|}\n")
    text_file.write("\\hline\n")
    text_file.write("$\\textrm{Atual}\diagdown\\textrm{Previsto}$ & \\textbf{positivo} & \\textbf{negativo}")
    if type>0:
        text_file.write(" & \\textbf{neutro}")
    if type>1:
        text_file.write(" & \\textbf{irrelevante}")
    text_file.write("\\\\ \\hline\n")
    text_file.write("\\textbf{positivo}")
    for i in range(2+type):
        text_file.write(" & %d" % metrics.confusion_matrix(testData.target, predicted)[0][i])
    text_file.write("\\\\ \\hline\n")
    text_file.write("\\textbf{negativo}")
    for i in range(2+type):
        text_file.write(" & %d" % metrics.confusion_matrix(testData.target, predicted)[1][i])
    text_file.write("\\\\ \\hline\n")
    if type>0:
        text_file.write("\\textbf{neutro}")
        for i in range(2+type):
            text_file.write(" & %d" % metrics.confusion_matrix(testData.target, predicted)[2][i])
        text_file.write("\\\\ \\hline\n")
    if type>1:
        text_file.write("\\textbf{irrelevante}")
        for i in range(2+type):
            text_file.write(" & %d" % metrics.confusion_matrix(testData.target, predicted)[3][i])
        text_file.write("\\\\ \\hline\n")
    text_file.write("\\end{tabular}\n")
    text_file.write("\\end{minipage}\n")
    text_file.write("\\hspace{0.5cm}\n")
    text_file.write("\\begin{minipage}[b]{0.45\linewidth}\n")
    text_file.write("\n")
    text_file.write("\\centering\n")
    text_file.write("\\caption{Medidas da Matriz de Confusão}\n")
    text_file.write("\\label{tab:mmcb-nb}\n")
    text_file.write("\\begin{tabular}{|l|l|l|l|}\n")
    text_file.write("\\hline\n")
    text_file.write("         & \\textbf{precisão} & \\textbf{recall} & \\textbf{f1-score} \\\\ \\hline\n")

    medidas = metrics.classification_report(testData.target, predicted, target_names=getCategory(type))
    listm = medidas.split()
    i=0
    while not listm[i]=="pos":
        i+=1
    text_file.write("\\textbf{positivo} & " + listm[i+1] + "     & " + listm[i+2] + "   & " + listm[i+3] + "     \\\\ \\hline\n")
    while not listm[i]=="neg":
        i+=1
    text_file.write("\\textbf{negativo} & " + listm[i+1] + "     & " + listm[i+2] + "   & " + listm[i+3] + "     \\\\ \\hline\n")
    if type>0:
        while not listm[i]=="neu":
            i+=1
        text_file.write("\\textbf{neutro} & " + listm[i+1] + "     & " + listm[i+2] + "   & " + listm[i+3] + "     \\\\ \\hline\n")
    if type>1:
        while not listm[i]=="irr":
            i+=1
        text_file.write("\\textbf{irrelevante} & " + listm[i+1] + "     & " + listm[i+2] + "   & " + listm[i+3] + "     \\\\ \\hline\n")
    while not listm[i]=="total":
        i+=1
    text_file.write("\\textbf{média} & " + listm[i+1] + "     & " + listm[i+2] + "   & " + listm[i+3] + "     \\\\ \\hline\n")

    text_file.write("\\textbf{acurácia} & \\multicolumn{3}{|c|}{%s}\\\\ \\hline\n" % metrics.accuracy_score(testData.target, predicted))
    text_file.write("\\end{tabular}\n")
    text_file.write("\\end{minipage}\n")
    text_file.write("\\end{table}")

    text_file.close()
