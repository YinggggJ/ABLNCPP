import codecs
import numpy as np

def EIIP_pos():
    f = codecs.open('features/lncFinder_pos_features.txt', mode='r', encoding='utf-8')
    line = f.readline()
    peak, SNR= [], []
    flag = 0
    while line:
        if flag == 0:
            line = f.readline()
            flag += 1
        else:
            feature = line.split()
            peak.append(float(feature[8]))
            SNR.append(float(feature[9]))
            line = f.readline()
    f.close()
    eiip_pos = [peak, SNR]
    eiip_pos = np.transpose(eiip_pos).tolist()
    return eiip_pos


def EIIP_neg():
    f = codecs.open('features/lncFinder_neg_features.txt', mode='r', encoding='utf-8')
    line = f.readline()
    peak, SNR = [], []
    flag = 0
    while line:
        if flag == 0:
            line = f.readline()
            flag += 1
        else:
            feature = line.split()
            peak.append(float(feature[8]))
            SNR.append(float(feature[9]))
            line = f.readline()
    f.close()
    eiip_neg = [peak, SNR]
    eiip_neg = np.transpose(eiip_neg).tolist()
    return eiip_neg
