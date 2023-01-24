import torch
import numpy as np
from sklearn.metrics import  classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from model.configs import FORM0_PATH, FORM1_PATH, FORM2_PATH, MODEL_PATH, DEVICE, CONFIG, LE_PATH, PART_NUM, VOCAB_PATH
from model.datasets import Datasets
from model.architecture import Model
from train import train
from model.utils import load_json, load_pkl
from model.evaluation import Evaluation


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def train_run():
    form0 = load_json(FORM0_PATH)
    form1 = load_json(FORM1_PATH)
    form2 = load_json(FORM2_PATH)
    forms = [s for s in zip(form0, form1, form2)]
    
    train_data, test_and_val_data = train_test_split(forms, test_size=0.2, random_state=42)
    test_data, val_data = train_test_split(test_and_val_data, test_size=0.5, random_state=42)

    train_datasets = Datasets(train_data, repeat=8)
    val_datasets = Datasets(val_data, repeat=8)

    train_loader = DataLoader(
        train_datasets,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_datasets,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )

    model = Model(
        vocab_size=len(load_json(VOCAB_PATH)),
        emb_dim=128,
        part_num=PART_NUM,
        features_num=7,
        outputs_size=len(load_pkl(LE_PATH).classes_),
    ).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
    criterion = torch.nn.CrossEntropyLoss()

    train(train_loader, val_loader, model, optimizer, criterion)


def test_run():
    form0 = load_json(FORM0_PATH)
    form1 = load_json(FORM1_PATH)
    form2 = load_json(FORM2_PATH)
    forms = [r for r in zip(form0, form1, form2)]
    train_data, test_and_val_data = train_test_split(forms, test_size=0.2, random_state=42)
    test_data, val_data = train_test_split(test_and_val_data, test_size=0.5, random_state=42)

    test_datasets = Datasets(test_data)

    test_loader = DataLoader(
        test_datasets,
        batch_size=1,
        shuffle=True,
        num_workers=1,
        drop_last=True,
    )

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

    y_true = []
    y_pred = []
    y_pred_score = []
    prob = []

    for idx, batch_data in enumerate(test_loader):
        inputs0,inputs1,inputs2, features, targets = batch_data
        outputs = model((inputs0.to(DEVICE),inputs1.to(DEVICE),inputs2.to(DEVICE)), features.to(DEVICE))

        y_true.extend(targets.reshape(-1).tolist())
        y_pred.extend(torch.argmax(outputs, dim=1).tolist())
        outputs = outputs.detach().cpu().tolist()
        prob.append(outputs[0])
    for i in range(len(prob)):
        prob[i] = np.array(prob[i])
        prob[i] = softmax(prob[i])
        y_pred_score.append(prob[i][1])
    Acc, Pre, Sn, Sp, F_score, MCC = Evaluation(y_true, y_pred)
    AUC = roc_auc_score(y_true, y_pred_score)
    print("\n Model Performance Evaluation: \n")
    print(classification_report(y_true, y_pred, target_names=load_pkl(LE_PATH).classes_, zero_division=0))
    print("Acc: {:.6f}  Pre: {:.6f} Sn: {:.6f}  Sp: {:.6f}  F_score: {:.6f}  MCC: {:.6f}  AUC: {:.6f}".format(Acc, Pre, Sn, Sp, F_score, MCC, AUC))


if __name__ == "__main__":
    train_run()
    test_run()
    

