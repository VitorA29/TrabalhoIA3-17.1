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
