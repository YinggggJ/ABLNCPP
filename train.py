import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from model.configs import MODEL_PATH, DEVICE, CONFIG


def train_epoch(train_loader, model, optimizer, criterion):
    model.train()
    train_loss_records = []
    for idx, batch_data in enumerate(tqdm(train_loader)):
        inputs0, inputs1, inputs2, features, targets = batch_data
        outputs = model((inputs0.to(DEVICE),inputs1.to(DEVICE),inputs2.to(DEVICE)), features.to(DEVICE))
        loss = criterion(outputs, targets.reshape(-1).to(DEVICE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss_records.append(loss.item())

    train_loss = round(sum(train_loss_records) / len(train_loss_records), 4)
    return train_loss


def evaluate(val_loader, model):
    model.eval()
    val_correct_num = []
    for idx, batch_data in enumerate(val_loader):
        inputs0, inputs1, inputs2, features, targets = batch_data
        outputs = model((inputs0.to(DEVICE),inputs1.to(DEVICE),inputs2.to(DEVICE)), features.to(DEVICE))
        val_correct_num.append(accuracy_score(targets.reshape(-1), torch.argmax(outputs, dim=1).cpu()))

    val_acc = round(sum(val_correct_num) / len(val_correct_num), 4)    
    return val_acc


def train(train_loader, val_loader, model, optimizer, criterion):
    best_val_acc = 0
    patience_counter = 0
    for epoch in range(1, CONFIG["epoch"] + 1):
        train_loss = train_epoch(train_loader, model, optimizer, criterion)
        print(f"[train]   Epoch: {epoch} / {CONFIG['epoch']}, Loss: {train_loss}")
        val_acc = evaluate(val_loader, model)
        print(f"[valid]   Epoch: {epoch} / {CONFIG['epoch']}, Acc: {val_acc}")

        if val_acc - best_val_acc > CONFIG["patience"]:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (patience_counter >= CONFIG["patience_num"] and epoch > CONFIG["min_epoch"]) or epoch == CONFIG["epoch"]:
            print(f"Best Val Acc: {best_val_acc}, Training Finished!")
            break
