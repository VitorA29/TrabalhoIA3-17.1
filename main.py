import sys
from Util import getClassificadoresQTD
from predict import predict

sucesso = False
while (len(sys.argv)==4):

    if (sys.argv[1]).upper() == "TRUE":
        gridSearch = True
    elif (sys.argv[1]).upper() == "FALSE":
        gridSearch = False
    else:
        print("gridSearch invalido\n")
        break

    if (sys.argv[2]).upper() == "TRUE":
        showWrongPredictions = True
    elif (sys.argv[2]).upper() == "FALSE":
        showWrongPredictions = False
    else:
        print("showWrongPredictions invalido\n")
        break

    if (sys.argv[3]).upper() == "TRUE":
        showPredictions = True
    elif (sys.argv[3]).upper() == "FALSE":
        showPredictions = False
    else:
        print("showPredictions invalido\n")
        break

    for j in range(3):
        for i in range(getClassificadoresQTD()):
            predict(i, j, gridSearch, showWrongPredictions, showPredictions)
    sucesso = True
    break
if sucesso == True:
    print("SUCCEFULL")
else:
    print("ERROR")
