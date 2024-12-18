import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from typing import List
from torch.optim import Adam

PAD_ID = 0
MASK_ID = 1
MAX_LEN = 64

def pad_and_mask(batch: List):
    np_batch = np.zeros((len(batch), MAX_LEN))
    for i in range(len(batch)):
        # Fill in row with the sequence, leaving the rest as 0
        np_batch[i, 0:min(len(batch[i]), MAX_LEN)] = batch[i][:MAX_LEN]
    masks = np_batch.astype(bool).astype(int)
    return np_batch, torch.tensor(masks)

# Probabilities here taken from the BERT paper, page 4
def add_mlm_masking(input_ids: np.ndarray, vocab_len: int):
    labels = deepcopy(input_ids)
    input_mask = np.random.choice([0, 1], size=(input_ids.shape), p=[0.15, 0.85])
    labels[input_mask == 1] = -100
    mlm_mask = np.random.choice([5, 6, 7], size=(input_ids.shape), p=[0.8, 0.1, 0.1]) # 5 for mask, 6 for random, and 7 for keep
    rdm = np.random.randint(2, vocab_len, size=(input_ids.shape)) # 2 to ignore PAD and MASK
    input_ids[(mlm_mask == 5) & (input_mask == 0)] = MASK_ID
    input_ids[(mlm_mask == 6) & (input_mask == 0)] = rdm[(mlm_mask == 6) & (input_mask == 0)]
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long) # for some reason, need to use torch.long for tensors coming from numpy


class ProcessedDataset(Dataset):
    def __init__(self, input_ids, masks, labels):
        self.input_ids = input_ids
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masks[idx], self.labels[idx]
    
    
def train(ids, mask, labels, val_ids, val_mask, val_labels, model, epochs=10, verbose=True):
    dataset = ProcessedDataset(ids, mask, labels)
    val = ProcessedDataset(val_ids, val_mask, val_labels)
    
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(val, batch_size=32, shuffle=True)
    
    # https://datascience.stackexchange.com/questions/64583/what-are-the-good-parameter-ranges-for-bert-hyperparameters-while-finetuning-it
    optimizer = Adam(model.parameters(), lr=2e-5) # huggingface doesn't seem to have regular adam

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device) 

    losses = []
    losses_val = []
    for epoch in range(epochs):
        total_loss = 0

        model.train()
        if verbose:
            itr = tqdm(dataloader)
        else:
            itr = dataloader
        for batch in itr:
            input_ids = batch[0].to(device)
            mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
            loss = outputs.loss # BertForMaskedLM calculates the loss for us

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            
        model.eval()
        total_loss_val = 0
        for batch in dataloader_val:
            input_ids = batch[0].to(device)
            mask = batch[1].to(device)
            labels = batch[2].to(device)
            with torch.no_grad():
                outputs = model(input_ids=input_ids, attention_mask=mask, labels=labels)
                loss = outputs.loss
                total_loss_val += loss.item()

        avg_loss = total_loss / len(dataloader)
        avg_loss_val = total_loss_val / len(dataloader_val)
        losses.append(avg_loss)
        losses_val.append(avg_loss_val)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {round(avg_loss, 3)}")
    
    return losses, losses_val

def get_p_diffs(losses_1, losses_2):
    assert len(losses_1) == len(losses_2), "Losses must have the same length"
    p_diffs = []
    for i in range(len(losses_1)):
        p_diffs.append((losses_2[i] - losses_1[i]) / losses_1[i] * 100)
    return p_diffs

def get_p_diff(loss_1, loss_2):
    return (loss_2 - loss_1) / loss_1 * 100







