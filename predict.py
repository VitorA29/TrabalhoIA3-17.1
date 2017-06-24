from sklearn import decomposition
from sklearn.linear_model import RandomizedLasso

from Util import *

def prediction():
    print("Reading files...")

    #Ele vai usar esse dataset pra aprender o que é negativo, positivo e neutro.
    train = file2SentencesArray('twitter-sanders-apple3')

    #Vai aplicar o conhecimento nesse.
    test = file2SentencesArray('twitter-sanders-apple2')
    print("Complete!")

    print("Cleaning sentences...")
    cleanTrainSentences = cleanSentences(train["text"])
    cleanTestSentences = cleanSentences(test["text"])
    print("Complete!...")
    # O trabalho começa aqui
    # Tem que usar as bibliotecas do enunciado(PCA e etc...) pra tratar a entrada e etc..
    print("Fiting sentences...")
    vectorizer = CountVectorizer(analyzer="word", tokenizer=None, preprocessor=None, stop_words=None, max_features=5000)
    trainDataFeatures = vectorizer.fit_transform(cleanTrainSentences)
    np.asarray(trainDataFeatures)

    #DANDO ERRO.

    #randomized_lasso = RandomizedLasso()
    #randomized_lasso.fit(trainDataFeatures, cleanTrainSentences)
    #trainDataFeatures = randomized_lasso.transform(trainDataFeatures)

    #pca = decomposition.PCA(n_components=2)
    #pca.fit_transform(trainDataFeatures)
    #trainDataFeatures = pca.transform(trainDataFeatures)

    testDataFeatures = vectorizer.transform(cleanTestSentences)
    np.asarray(testDataFeatures)
    print("Complete!")

    #E aqui usar diferentes classificadores, o RandomForest é um deles.
    print("Predicting...")
    forest = RandomForestClassifier(n_estimators=100)
    forest = forest.fit(trainDataFeatures, train["class"])
    result = forest.predict(testDataFeatures)
    print("Complete...")

    return result

def writeResult(resultArray):
    print(resultArray)

#writeResult(prediction())

prediction()