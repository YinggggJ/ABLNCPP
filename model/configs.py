import os
import torch

def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)

# root
ROOT_DIR = os.path.abspath(".")

# data
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "data.csv")
FORM0_PATH = os.path.join(DATA_DIR, "form0_embed.json")
FORM1_PATH = os.path.join(DATA_DIR, "form1_embed.json")
FORM2_PATH = os.path.join(DATA_DIR, "form2_embed.json")

# model
OUTPUTS_DIR = os.path.join(ROOT_DIR, "outputs")
VOCAB_PATH = os.path.join(OUTPUTS_DIR, "vocab.json")
LE_PATH = os.path.join(OUTPUTS_DIR, "le.pkl")
MODEL_PATH = os.path.join(OUTPUTS_DIR, "ABLNCPP.pkl")

TOKEN_LEN = 3 
PART_NUM = 64
PART_LEN = 100

# remkdir
makedir(OUTPUTS_DIR)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# model configs
CONFIG = {
    "batch_size": 128,
    "lr": 0.001,
    "epoch": 50,
    "min_epoch": 5,
    "patience": 0.0002,  # early stopping
    "patience_num": 10,
}
