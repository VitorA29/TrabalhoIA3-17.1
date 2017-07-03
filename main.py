import sys
from Util import getClassificadoresQTD, BINARIO, TERNARIO, QUATERNARIO
from predict import predict

sucesso = False
while (len(sys.argv)==8):

    classificador = int(sys.argv[1])
    if not (classificador<getClassificadoresQTD() and classificador>=0):
        print("Classificador invalido\n")
        break

    if (sys.argv[2]).upper() == "BINARIO":
        type = BINARIO
    elif (sys.argv[2]).upper() == "TERNARIO":
        type = TERNARIO
    elif (sys.argv[2]).upper() == "QUATERNARIO":
        type = QUATERNARIO
    else:
        print("Modo invalido\n")
        break

    if (sys.argv[3]).upper() == "TRUE":
        gridSearch = True
    elif (sys.argv[3]).upper() == "FALSE":
        gridSearch = False
    else:
        print("gridSearch invalido\n")
        break

    if (sys.argv[4]).upper() == "TRUE":
        showWrongPredictions = True
    elif (sys.argv[4]).upper() == "FALSE":
        showWrongPredictions = False
    else:
        print("showWrongPredictions invalido\n")
        break

    if (sys.argv[5]).upper() == "TRUE":
        showPredictions = True
    elif (sys.argv[5]).upper() == "FALSE":
        showPredictions = False
    else:
        print("showPredictions invalido\n")
        break

    if (sys.argv[6]).upper() == "TRUE":
        rfeEnabled = True
    elif (sys.argv[6]).upper() == "FALSE":
        rfeEnabled = False
    else:
        print("rfeEnabled invalido\n")
        break

    if (sys.argv[7]).upper() == "TRUE":
        pcaEnabled = True
    elif (sys.argv[7]).upper() == "FALSE":
        pcaEnabled = False
    else:
        print("pcaEnabled invalido\n")
        break

    predict(classificador, type, gridSearch, showWrongPredictions, showPredictions, rfeEnabled, pcaEnabled)

    sucesso = True
    break
if sucesso == True:
    print("SUCCEFULL")
else:
    print("ERROR")
