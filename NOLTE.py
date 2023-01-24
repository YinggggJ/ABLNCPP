import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from model.configs import DATA_PATH, VOCAB_PATH, FORM0_PATH, FORM1_PATH, FORM2_PATH, LE_PATH, TOKEN_LEN, PART_NUM, PART_LEN
from model.utils import save_pkl, save_json, load_json


def label_encoder(y, label_model_path):
    le = LabelEncoder()
    labels = le.fit_transform(y)
    save_pkl(label_model_path, le)
    return le, labels


class Vocabulary(object):
    def __init__(self):
        self.token2idx = {"<K>": 0}
        self.idx2token = {0: "<K>"}
        self.idx = 1

    def add_token(self, token):
        if token not in self.token2idx:
            self.token2idx[token] = self.idx
            self.idx2token[self.idx] = token
            self.idx += 1

    def add_seq(self, seq):
        for token in seq.split():
            self.add_token(token)

    def seq2vec(self, seq):
        seq_idxs = [self.token2idx.get(token, self.token2idx["<K>"]) for token in seq.split()]
        return seq_idxs

    def save_dict(self, dict_path):
        save_json(dict_path, self.token2idx)

    def load_dict(self, dict_path):
        self.token2idx = load_json(dict_path)
        self.idx2token = {idx: token for token, idx in self.token2idx.items()}
        self.idx = len(self.token2idx)

    def __call__(self, token):
        return self.token2idx.get(token, self.token2idx["<K>"])

    def __len__(self):
        return self.idx


if __name__ == "__main__":
    data = pd.read_csv(DATA_PATH)
    vocabulary_obj = Vocabulary()

    # build vocabulary(corpus) with k-mer algorithm
    for seq in tqdm(data["seq"].tolist()):
        seq_slide = " ".join(
            [seq[start_idx : start_idx + TOKEN_LEN] for start_idx in range(0, len(seq) - TOKEN_LEN + 1)]
        )
        vocabulary_obj.add_seq(seq_slide)
    vocabulary_obj.save_dict(VOCAB_PATH)

    le, labels = label_encoder(data["labels"].tolist(), LE_PATH)

    # three forms of one ncRNA sequence
    form0 = []
    for seq, features, label in tqdm(
        zip(data["seq"].tolist(), data["features"].tolist(), labels.tolist()), total=len(data)
    ):
        seq = (
            seq[: PART_NUM * PART_LEN]
            if len(seq) >= PART_NUM * PART_LEN
            else seq + "#" * (PART_NUM * PART_LEN - len(seq))
        )
        data_info0 = {}
        # transform sequence into vector
        for part_start_idx in range(0, PART_NUM):
            part_seq = seq[part_start_idx * PART_LEN : (part_start_idx + 1) * PART_LEN]
            part_seq_slide = " ".join(
                [
                    part_seq[token_start_idx : token_start_idx + TOKEN_LEN]
                    for token_start_idx in range(0, len(part_seq) - TOKEN_LEN + 1, 3)
                ]
            )
            data_info0[f"#{part_start_idx}"] = vocabulary_obj.seq2vec(part_seq_slide)
        data_info0["features"] = eval(features)
        data_info0["label"] = label
        form0.append(data_info0)
    save_json(FORM0_PATH, form0)

    form1 = []
    for seq, features, label in tqdm(
        zip(data["seq"].tolist(), data["features"].tolist(), labels.tolist()), total=len(data)
    ):
        seq = (
            seq[: PART_NUM * PART_LEN]
            if len(seq) >= PART_NUM * PART_LEN
            else seq + "#" * (PART_NUM * PART_LEN - len(seq))
        )
        data_info1 = {}
        # transform sequence into vector
        for part_start_idx in range(0, PART_NUM):
            part_seq = seq[part_start_idx * PART_LEN : (part_start_idx + 1) * PART_LEN]
            part_seq_slide = " ".join(
                [
                    part_seq[token_start_idx : token_start_idx + TOKEN_LEN]
                    for token_start_idx in range(1, len(part_seq) - TOKEN_LEN + 1, 3)
                ]
            )
            data_info1[f"#{part_start_idx}"] = vocabulary_obj.seq2vec(part_seq_slide)
        data_info1["features"] = eval(features)
        data_info1["label"] = label
        form1.append(data_info1)
    save_json(FORM1_PATH, form1)

    form2 = []
    for seq, features, label in tqdm(
        zip(data["seq"].tolist(), data["features"].tolist(), labels.tolist()), total=len(data)
    ):
        seq = (
            seq[: PART_NUM * PART_LEN]
            if len(seq) >= PART_NUM * PART_LEN
            else seq + "#" * (PART_NUM * PART_LEN - len(seq))
        )
        data_info2 = {}
        # transform sequence into vector
        for part_start_idx in range(0, PART_NUM):
            part_seq = seq[part_start_idx * PART_LEN : (part_start_idx + 1) * PART_LEN]
            part_seq_slide = " ".join(
                [
                    part_seq[token_start_idx : token_start_idx + TOKEN_LEN]
                    for token_start_idx in range(2, len(part_seq) - TOKEN_LEN + 1, 3)
                ]
            )
            data_info2[f"#{part_start_idx}"] = vocabulary_obj.seq2vec(part_seq_slide)
        data_info2["features"] = eval(features)
        data_info2["label"] = label
        form2.append(data_info2)
    save_json(FORM2_PATH, form2)