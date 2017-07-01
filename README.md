# Terceiro trabalho da disciplina de Inteligência Artificial 2017.1.

## Usar ferramentas de Machine Learning para Analise de Sentimento.

## Grupo
- Max Fratane.
- Erick Grilo.
- Vitor Araújo.
- Vitor Lourenço.

## Dataset
São usados Tweets para treinar e para testar o aprendizado.
A pasta `train` contém todos os arquivos que serão usados para treinamento, e a pasta `test` contém todos que serão usados para teste.
Juntando todos os arquivos em cada pasta, é um total de, aproximadamente, 27000 Tweets de treinamento e 900 de teste.

## QuickStart

O pack de stopwords utilizado é baixado atraves do comando `nltk.download()`, caso não tenha, execute o comando antes das chamadas das funções do arquivo `predict.py`.

Dentro do `predict.py`, executar a função `def predict(classifier, type, gridSearch, showWrongPredict, showPredictions):`

por exemplo:

``` python
type = QUATERNARIO
gridSearch = False
showWrongPredictions = True
showPredictions = False
print(predict(RANDOM_FOREST, type, gridSearch, showWrongPredictions, showPredictions))
```

Onde,
`type` é a quantidade de tipos de tweets. O Quaternário tem, pro exemplo, Positivo, Negativo, Neutro e Irrelevante.
`gridSearch` é se vai utilizar o gridSearch ou não.
`showWrongPredictions` Mostrar ou não as previsões erradas do classificador.
`showPredictions` Mostrar ou não todas as predições.

`DECISION_TREE`, `NAIVE_BAYES`, `RANDOM_FOREST`, `SVM` são as constantes que representam os classificadores.

### Módulos usados
`SciPy`
`NumPy`
`Sklearn`

### Confusion Matrix
|     | Pos | Neg | Neu | Irr |
|-----|-----|-----|-----|-----|
| Pos | 163 | 0   | 0   | 0   |
| Neg | 0   | 316 | 0   | 0   |
| Neu | 7   | 0   | 502 | 0   |
| Irr | 19  | 9   | 51  | 6   |

Nessa exemplo, vemos que a maior confusão ocorre na categoria Irregular. Ela é muito confundida com Neutro e Positivo.
Mas no geral, o resultado é muito bom.

### Util.py
Contem funções e classes utilitárias.

``` python
'''
  Classe que agrupa todos os tweets e a classificação de cada um deles.
  O atributo Text armazena o tweet e classific sua classificação.
  Ambos são vetores e o classific[0] corresponde a classificação do text[0].
'''
class Data(object)
```

``` python
'''
  Retorna uma instancia de Data, contendo todos os tweets, e suas as classificações, de todos os arquivos da pasta test.
'''
def getTestData()
```

``` python
'''
  Retorna uma instancia de Data, contendo todos os tweets, e suas as classificações, de todos os arquivos da pasta train.
'''
def getTrainData():
```

``` python
'''
    Dado um arquivo .csv, joga todos elementos pra um array.
    :param fileName: Nome do arquivo sem a extenção.
    :return: retorna um array de comentário e se sua classificação.
'''
def file2SentencesArray(fileName, folderName)
```

``` python
'''
    Dado um texto, transforma em um array de palavras.
    :param text: text a ser separado em palavras.
    :param removeStopwords: True caso queria que as stopwords sejam removidas, False(padrão) caso contrário.
    :return: array de palavras.
'''
def text2Wordlist(text, removeStopwords = False)
```

``` python
    '''
    Limpa uma frase de caracteres insedejados.
    :param sentencesArray: Array contendo frases.
    :return: retorna um array de frases sem certos caracteres.
    '''
def cleanSentences(sentencesArray)
```

``` python
'''
    Lista todos os arquivos de um caminho.
'''
def list_files(path)
```
### Predict.py


``` python
'''
    Realiza as predições.
    Usando as funções utilitárias, lê todos os arquivos de teste e train, joga tudo para duas instancias de Data,
    uma contendo os dados de teste e outra de treinamento.
    
    Printa alguns relatórios e estatísticas, predições dos arquivos de teste e a confusion matrix.
'''
def predict()
```
