import math

def Evaluation(y_true, y_pred):
    TP, TN, FN, FP = 0, 0, 0, 0

    for i, j in zip(y_true, y_pred):
        if j == i and i == 1:
            TP += 1
        elif j == i and i == 0:
            TN += 1
        elif j != i and i == 1:
            FN += 1
        elif j != i and i == 0:
            FP += 1

    ACC = (TP + TN) / (TP + TN + FN + FP)
    PRE = TP / (TP + FP)
    SN = TP / (TP + FN)
    SP = TN / (TN + FP)
    F_score = 2 * PRE * SN / (PRE + SN)
    MCC = (TP * TN - FP * FN) / math.sqrt((TP + FN) * (TP + FP) * (TN + FP) * (TN + FN))
    
    return ACC, PRE, SN, SP, F_score, MCC
