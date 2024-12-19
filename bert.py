import torch
from torch.utils.data import Dataset, DataLoader
from copy import deepcopy
from tqdm import tqdm
import numpy as np
from torch.optim import Adam

"""
----------------------------------------------------------------------------------------------------------
This file contains the methods related to preparing data for MLM and training BERT models for experiments.
----------------------------------------------------------------------------------------------------------
"""

# Set the ids for special tokens
PAD_ID = 0
MASK_ID = 1
MAX_LEN = 64

def pad_and_mask(input_ids):
    """
    Pad each input up to the max length and creates the masks for the MLM task.
    
    Args:
        input_ids: 2D array of tokens

    Returns:
        Tuple of padded inputs and masks
    """
    # Create the initial NP array that will store padded sequences
    np_inputs = np.zeros((len(input_ids), MAX_LEN))
    for i in range(len(input_ids)):
        # Fill in row with the sequence, leaving the rest as 0
        np_inputs[i, 0:min(len(input_ids[i]), MAX_LEN)] = input_ids[i][:MAX_LEN]
    # Create a mask that indicates which tokens are padding and which are not
    masks = np_inputs.astype(bool).astype(int)
    # Returning an NP array for the inputs as it needs to be used in add_mlm_masking()
    return np_inputs, torch.tensor(masks)


def add_mlm_masking(input_ids: np.ndarray, vocab_len: int):
    """
    Adjust the inputs for the MLM task, replacing some tokens with the mask/random token
    
    Args:
        input_ids: 2D array of tokens. Should already be padded.

    Returns:
        Tuple of the updaated inputs and mlm labels
    """
    # Probabilities here taken from the BERT paper, page 4
    labels = deepcopy(input_ids)
    # 15% of tokens will be replaced
    try:
        input_mask = np.random.choice([0, 1], size=(input_ids.shape), p=[0.15, 0.85])
    except Exception as e:
        raise Exception(f"Error creating input mask. Make sure input_ids is an np.array from pad_and_mask(): {e}")
    # No predictions need to be made for tokens that aren't replaced
    labels[input_mask == 1] = -100
    # 80% mask, 10% random token, 10% same token
    mlm_mask = np.random.choice([5, 6, 7], size=(input_ids.shape), p=[0.8, 0.1, 0.1]) # 5 for mask, 6 for random, and 7 for keep
    # Choose the random tokens
    rdm = np.random.randint(2, vocab_len, size=(input_ids.shape)) # 2 to ignore PAD and MASK
    # Replace the correct input_ids with the mask value and the random tokens. Tokens that are the same don't need to be changed.
    input_ids[(mlm_mask == 5) & (input_mask == 0)] = MASK_ID
    input_ids[(mlm_mask == 6) & (input_mask == 0)] = rdm[(mlm_mask == 6) & (input_mask == 0)]
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long) # for some reason, need to use torch.long for tensors coming from numpy


class ProcessedDataset(Dataset):
    """Simple dataset for the MLM task"""
    def __init__(self, input_ids, masks, labels):
        """
        Args:
            input_ids: 2D array of tokens. Should already be padded and masked.
            masks: 2D array of masks indicating which tokens are padding
            labels: 2D array of mlm labels
        """
        self.input_ids = input_ids
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.masks[idx], self.labels[idx]
    
    
def train(ids, mask, labels, val_ids, val_mask, val_labels, model, epochs=10, verbose=True):
    """
    Trains BERT using the MLM task.
    
    Args:
        ids: 2D array of training tokens. Should already be padded and masked.
        mask: 2D array of training masks indicating which tokens are padding
        labels: 2D array of training mlm labels
        val_ids: 2D array of validation tokens. Should already be padded and masked.
        val_mask: 2D array of validation masks indicating which tokens are padding
        val_labels: 2D array of validation mlm labels
        model: BertForMaskedLM model
        epochs: Number of epochs to run
        verbose: Show progress bars and loss updates.
        
    Returns:
        Tuple of the training and validation loss
    """
    
    # Load the datasets
    dataset = ProcessedDataset(ids, mask, labels)
    val = ProcessedDataset(val_ids, val_mask, val_labels)
    
    # Create the dataloaders
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    dataloader_val = DataLoader(val, batch_size=32, shuffle=True)
    
    # https://datascience.stackexchange.com/questions/64583/what-are-the-good-parameter-ranges-for-bert-hyperparameters-while-finetuning-it
    optimizer = Adam(model.parameters(), lr=2e-5) # huggingface doesn't seem to have regular adam

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    model.to(device) 

    # Run the training, keeping track of train and val loss
    losses = []
    losses_val = []
    for epoch in range(epochs):
        total_loss = 0

        model.train()
        # Change the iterator used in the loop depending on the verbose parameeter
        if verbose:
            itr = tqdm(dataloader)
        else:
            itr = dataloader
        
        # Run a batch
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
        
        # Test the model on the validation set without changing the model
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

        # Caluclate the losses for the epoch
        avg_loss = total_loss / len(dataloader)
        avg_loss_val = total_loss_val / len(dataloader_val)
        losses.append(avg_loss)
        losses_val.append(avg_loss_val)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {round(avg_loss, 3)}")
    
    # Just returns the losses found, which is helpful for the experiments
    return losses, losses_val

def get_p_diffs(losses_1, losses_2):
    """
    Gets the percent differences between lists of losses
    
    Args:
        losses_1: "New" list of losses
        losses_1: "Old" list of losses
        
    Returns:
        List of percent difference in loss at each point
    """
    # Got a out of bounds errors when writing experiments, so this is assertion is helpful
    assert len(losses_1) == len(losses_2), "Losses must have the same length"
    p_diffs = []
    # Calculate the percent difference for each loss pair
    for i in range(len(losses_1)):
        p_diffs.append((losses_2[i] - losses_1[i]) / losses_1[i] * 100)
    return p_diffs

def get_p_diff(loss_1, loss_2):
    """
    Gets the percent differences between 2 loss values
    
    Args:
        loss_1: "New" loss value
        loss_2: "Old" loss value
        
    Returns:
        The percent difference in loss
    """
    # Simple percent difference calculation
    return (loss_2 - loss_1) / loss_1 * 100







