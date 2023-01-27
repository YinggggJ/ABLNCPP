import pandas as pd
from tqdm import tqdm
import argparse

from model.configs import DATA_PATH
from features import FrameKmer, Fickett, ORF_Length, EIIP


hex_file = "features/ncRNA_Human_Hexamer.tsv"
coding, noncoding = FrameKmer.coding_nocoding_potential(hex_file)

# preprocess fasta format data to extract sequences
def read_data(data_path):
    with open(data_path, "r", encoding="utf-8") as fr:
        data = []
        seq = ""
        for idx, row in enumerate(fr):
            if row.startswith(">"):
                if seq != "":
                    data.append(seq)
                seq = ""
            else:
                seq += row.strip("\n")
        if seq != "":
            data.append(seq)
        return data

# calculate biological features
def bio_features(seq):
    features = []
    features.append(FrameKmer.kmer_ratio(seq, 6, 3, coding, noncoding))
    features.append(Fickett.fickett_value(seq))
    Length, Coverage, Integrity = ORF_Length.len_cov(seq)
    features.append(Length)
    features.append(Coverage)
    features.append(Integrity)
    return features

def get_features(seqs):
    features = []
    for seq in tqdm(seqs):
        feature = bio_features(seq)
        features.append(feature)
    return features


parser = argparse.ArgumentParser()
parser.add_argument('--withcp', type=str, default = None)    #with coding potential file
parser.add_argument('--withoutcp', type=str, default = None)    #without coding potential file
args = parser.parse_args()
POSITIVE_DATA = args.withcp
NEGATIVE_DATA = args.withoutcp

if __name__ == "__main__":
    print("Data preprocessing:")
    positive_data = read_data(POSITIVE_DATA)
    positive_features = get_features(positive_data)
    # properties of ncRNA generated by LncFinder
    positive_eiip = EIIP.EIIP_pos()
    for i in range(len(positive_features)):
        positive_features[i] += positive_eiip[i]
    positive_data = pd.DataFrame(
        {
            "seq": positive_data,
            "features": positive_features,
            "labels": ["positive"] * len(positive_data),
        }
    )

    negative_data = read_data(NEGATIVE_DATA)
    negative_features = get_features(negative_data)
    negative_eiip = EIIP.EIIP_neg()
    for i in range(len(negative_features)):
        negative_features[i] += negative_eiip[i]
    negative_data = pd.DataFrame(
        {
            "seq": negative_data,
            "features": negative_features,
            "labels": ["negative"] * len(negative_data),
        }
    )

    data = pd.concat((positive_data, negative_data)).reset_index(drop=True)
    data.to_csv(DATA_PATH, index=False)