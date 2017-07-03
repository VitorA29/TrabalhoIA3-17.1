import sys
import shutil
from Util import getClassificadoresQTD, SVM
from predict import predict

sucesso = False
while (len(sys.argv)==3):

    if (sys.argv[1]).upper() == "TRUE":
        showWrongPredictions = True
    elif (sys.argv[1]).upper() == "FALSE":
        showWrongPredictions = False
    else:
        print("showWrongPredictions invalido\n")
        break

    if (sys.argv[2]).upper() == "TRUE":
        showPredictions = True
    elif (sys.argv[2]).upper() == "FALSE":
        showPredictions = False
    else:
        print("showPredictions invalido\n")
        break

    shutil.rmtree("execucao/")

    for j in range(3):
        for i in range(getClassificadoresQTD()):
            predict(i, j, False, showWrongPredictions, showPredictions, False, False)
            #if(i==SVM):
                #predict(i, j, True, showWrongPredictions, showPredictions, False, False)
                #predict(i, j, False, showWrongPredictions, showPredictions, True, False)
                #predict(i, j, False, showWrongPredictions, showPredictions, False, True)
    sucesso = True
    break
if sucesso == True:
    print("SUCCEFULL")
else:
    print("ERROR")
