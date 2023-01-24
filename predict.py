import torch

from model.configs import VOCAB_PATH, PART_NUM, LE_PATH, DEVICE, MODEL_PATH, PART_LEN, TOKEN_LEN
from features import FrameKmer, Fickett, ORF_Length
from model.architecture import Model
from model.utils import load_json, load_pkl
from NOLTE import Vocabulary
import numpy as np

hex_file = "features/ncRNA_Human_Hexamer.tsv"
coding, noncoding = FrameKmer.coding_nocoding_potential(hex_file)

le = load_pkl(LE_PATH)
vocabulary_obj = Vocabulary()
vocabulary_obj.load_dict(VOCAB_PATH)

model = (
    Model(
        vocab_size=len(load_json(VOCAB_PATH)),
        emb_dim=128,
        part_num=PART_NUM,
        features_num=7,
        outputs_size=len(load_pkl(LE_PATH).classes_),
    )
    .to(DEVICE)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def build_inputs(seq):
    seq = (
        seq[: PART_NUM * PART_LEN] if len(seq) >= PART_NUM * PART_LEN else seq + "#" * (PART_NUM * PART_LEN - len(seq))
    )
    data_info0 = {}
    for part_start_idx in range(0, PART_NUM):
        part_seq = seq[part_start_idx * PART_LEN : (part_start_idx + 1) * PART_LEN]
        part_seq_slide = " ".join(
            [
                part_seq[token_start_idx : token_start_idx + TOKEN_LEN]
                for token_start_idx in range(0, len(part_seq) - TOKEN_LEN + 1, 3)
            ]
        )
        data_info0[f"#{part_start_idx}"] = vocabulary_obj.seq2vec(part_seq_slide)

    data_info1 = {}
    for part_start_idx in range(0, PART_NUM):
        part_seq = seq[part_start_idx * PART_LEN : (part_start_idx + 1) * PART_LEN]
        part_seq_slide = " ".join(
            [
                part_seq[token_start_idx : token_start_idx + TOKEN_LEN]
                for token_start_idx in range(1, len(part_seq) - TOKEN_LEN + 1, 3)
            ]
        )
        data_info1[f"#{part_start_idx}"] = vocabulary_obj.seq2vec(part_seq_slide)

    data_info2 = {}
    for part_start_idx in range(0, PART_NUM):
        part_seq = seq[part_start_idx * PART_LEN : (part_start_idx + 1) * PART_LEN]
        part_seq_slide = " ".join(
            [
                part_seq[token_start_idx : token_start_idx + TOKEN_LEN]
                for token_start_idx in range(2, len(part_seq) - TOKEN_LEN + 1, 3)
            ]
        )
        data_info2[f"#{part_start_idx}"] = vocabulary_obj.seq2vec(part_seq_slide)
    inputs0 = torch.LongTensor([[data_info0[f"#{seq_idx}"] for seq_idx in range(PART_NUM)]])
    inputs1 = torch.LongTensor([[data_info1[f"#{seq_idx}"] for seq_idx in range(PART_NUM)]])
    inputs2 = torch.LongTensor([[data_info2[f"#{seq_idx}"] for seq_idx in range(PART_NUM)]])  
    return inputs0, inputs1, inputs2

def bio_features(seq):
    features = []
    features.append(FrameKmer.kmer_ratio(seq, 6, 3, coding, noncoding))
    features.append(Fickett.fickett_value(seq))
    Length, Coverage, Integrity = ORF_Length.len_cov(seq)
    features.append(Length)
    features.append(Coverage)
    features.append(Integrity)
    return features

if __name__ == "__main__":
    seq = "GCTGAGTCATCACTAGAGAGTGGGAAGGGCAGCAGCAGCAGAGAATCCAAACCCTAAAGCTGATATCACAAAGTACCATTTCTCCAAGTTGGGGGCTCAGAGGGGAGTCATCATGAGCGATGTTACCATTGTGAAAGAAGGTTGGGTTCAGAAGAGGGGAGAATATATAAAAAACTGGAGGCCAAGATACTTCCTTTTGAAGACAGATGGCTCATTCATAGGATATAAAGAGAAACCTCAAGATGTGGATTTACCTTATCCCCTCAACAACTTTTCAGTGGCAAAATGCCAGTTAATGAAAACAGAACGACCAAAGCCAAACACATTTATAATCAGATGTCTCCAGTGGACTACTGTTATAGAGAGAACATTTCATGTAGATACTCCAGAGGAAAGGGAAGAATGGACAGAAGCTATCCAGGCTGTAGCAGACAGACTGCAGAGGCAAGAAGAGGAGAGAATGAATTGTAGTCCAACTTCACAAATTGATAATATAGGAGAGGAAGAGATGGATGCCTCTACAACCCATCATAAAAGAAAGACAATGAATGATTTTGACTATTTGAAACTACTAGGTAAAGGCACTTTTGGGAAAGTTATTTTGGTTCGAGAGAAGGCAAGTGGAAAATACTATGCTATGAAGATTCTGAAGAAAGAAGTCATTATTGCAAAGGATGAAGTGGCACACACTCTAACTGAAAGCAGAGTATTAAAGAACACTAGACATCCCTTTTTAACATCCTTGAAATATTCCTTCCAGACAAAAGACCGTTTGTGTTTTGTGATGGAATATGTTAATGGGGGCGAGCTGTTTTTCCATTTGTCGAGAGAGCGGGTGTTCTCTGAGGACCGCACACGTTTCTATGGTGCAGAAATTGTCTCTGCCTTGGACTATCTACATTCCGGAAAGATTGTGTACCGTGATCTCAAGTTGGAGAATCTAATGCTGGACAAAGATGGCCACATAAAAATTACAGATTTTGGACTTTGCAAAGAAGGGATCACAGATGCAGCCACCATGAAGACATTCTGTGGCACTCCAGAATATCTGGCACCAGAGGTGTTAGAAGATAATGACTATGGCCGAGCAGTAGACTGGTGGGGCCTAGGGGTTGTCATGTATGAAATGATGTGTGGGAGGTTACCTTTCTACAACCAGGACCATGAGAAACTTTTTGAATTAATATTAATGGAAGACATTAAATTTCCTCGAACACTCTCTTCAGATGCAAAATCATTGCTTTCAGGGCTCTTGATAAAGGATCCAAATAAACGCCTTGGTGGAGGACCAGATGATGCAAAAGAAATTATGAGACACAGTTTCTTCTCTGGAGTAAACTGGCAAGATGTATATGATAAAAAGCTTGTACCTCCTTTTAAACCTCAAGTAACATCTGAGACAGATACTAGATATTTTGATGAAGAATTTACAGCTCAGACTATTACAATAACACCACCTGAAAAATGTCAGCAATCAGATTGTGGCATGCTGGGTAACTGGAAAAAATAATAAAAATCGGCTTCCTACAGCCAGCAGCACAGTCACCCATGGAACTGTTGGCTTTGGATTAAATGTGGAATTGAACGACTACCCAGAAGTGTTCTGGAAAGAAGCGAGATGTGTGGCCTGCCTCACCGTCCTCACCCATCAAAAGCACCAGCAGGCACGTTAACTCGAATTCTCACAAGGAAAAGGCCATTAAAGCTCAAGGTGCATTTCAAACTCCAGGCTAC"
    #biological features
    features = bio_features(seq)
    # Peak and SNR 
    features.append(8.57999493)
    features.append(0.339276017268937)

    inputs0, inputs1, inputs2 = build_inputs(seq)
    outputs = model((inputs0.cuda(), inputs1.cuda(), inputs2.cuda()), torch.FloatTensor([features]).cuda())
    
    prob = softmax(outputs.cpu().detach().numpy()).tolist()
    prob = prob[0][1]
    if prob > 0.5:
        print("Predict Result : The predicted ncRNA has coding potential.")
    else:
        print("Predict Result : The predicted ncRNA has no coding potential.")
    print(f"Coding Potential : {prob}")
