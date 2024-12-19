from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast
from tqdm import tqdm
from datasets import load_dataset
import pickle
import torch
import faiss
import numpy as np
from copy import deepcopy
from utils import copytree, create_dir, delete_dir
import os
import json
import time

"""
-----------------------------------------------------------------------------------------------
This file contains the methods related to collapsing the language and creating the new dataset.
-----------------------------------------------------------------------------------------------
"""

def get_bert_tokenizer_vocab(dataset, bert_tokenizer):
    """
    Collect the BERT tokenizer vocabulary and save it. Deprecated.
    
    Args:
        dataset: The text dataset to tokenizer
        bert_tokenizer: BERT's tokenizer from "google-bert/bert-base-uncased"
    
    Returns:
        The list of tokens in the vocabulary.
    """
    token_set = set()
    # Loop through each sample
    for i in tqdm(range(len(dataset))):
        # Extract the text and tokenize it
        story = dataset[i]['text']
        tokenized_story = bert_tokenizer.tokenize(story, return_tensors='pt', padding=False)
        # Keep track of tokens we have seen
        token_set.update(tokenized_story)
    # Remove special tokens
    token_set = token_set - set(bert_tokenizer.all_special_tokens) # For UNK in particular
    # Save a file containing the list found
    with open("token_list.pkl", "wb") as file:
        pickle.dump(list(token_set), file)
    return list(token_set)


# https://stackoverflow.com/questions/69780823/tokenizers-change-vocabulary-entry
def build_starting_bert_tokenizer():
    """
    Create a new BERT tokenizer using the tokens from the dataset. Deprecated.
    """
    # Load the token set
    try:
        with open("token_list.pkl", "rb") as file:
            token_list = pickle.load(file)
    except Exception as e:
        raise Exception(f"Couldn't load token list. Make sure you have run get_bert_tokenizer_vocab(): {e}")
    
    # Prepare the tokens for saving into tokenizer files
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    token_list = special_tokens + token_list
    vocab = {}
    for i in range(len(token_list)):
        vocab[token_list[i]] = i
    
    # Make the initial directory
    try:
        create_dir("bert_filtered_tokenizer", delete_existing=True)
        copytree("bert_tokenizer", "bert_filtered_tokenizer")
    except Exception as e:
        raise Exception(f"Couldn't copy the tokenizer. Make sure you have the bert_tokenizer folder from HuggingFace: {e}")
    
    # Edit the vocab.json file
    with open("bert_filtered_tokenizer/vocab.txt", "w") as file:
        file.write("\n".join(token_list))
        
    # Edit the tokenizer.json file
    with open("bert_filtered_tokenizer/tokenizer.json", "r", encoding='utf-8') as file:
        tokenizer_info = json.load(file)
    tokenizer_info['model']['vocab'] = vocab
    for special in tokenizer_info['added_tokens']:
        special['id'] = vocab[special['content']]
    for special in tokenizer_info['post_processor']['special_tokens']:
        info = tokenizer_info['post_processor']['special_tokens'][special]
        info["ids"] = [vocab[info["id"]]]
    tokenizer_info['normalizer']['handle_chinese_chars'] = False
    # Save the updated tokenizer file
    with open("bert_filtered_tokenizer/tokenizer.json", "w") as file:
        json.dump(tokenizer_info, file)
        

def get_collapsed_words(collapse_factor, tokenizer, original_embedding_layer, search_reduction_factor=25, use_saved=False, ids_to_keep=[]):
    """
    Get the collapsed word mappings from a model's tokenizer and embedding layer.
    
    Args:
        collapse_factor: The collapse factor that determines how much to reduce the language by.
        tokenizer: The original tokenizer used by the original model.
        original_embedding_layer: The embedding layer from the original model.
        search_reduction_factor: Used to reduce the number of similar vectors to find for each search. Setting this to 1 is an exhasutive search, but requires a decent amount of memory.
        use_saved: Whether or not to return saved files for the collapse mappings.
        ids_to_keep: list of indexes/IDs to not collapse
    
    Returns:
        A tuple of the mappings in the form of 'token to new id' and 'id to new id'
    """
    # Use the saved mappings if they exist
    if use_saved and os.path.exists("token_to_new_id.pkl") and os.path.exists("id_to_new_id.pkl"):
        try:
            with open("token_to_new_id.pkl", "rb") as file:
                token_to_new_id = pickle.load(file)
            with open("id_to_new_id.pkl", "rb") as file:
                id_to_new_id = pickle.load(file)
            return token_to_new_id, id_to_new_id
        except Exception as e:
            raise Exception(f"Couldn't load mappings. Try running the function use_saved=False: {e}")
    
    # Get the initial vocabulary 
    token_list = tokenizer.get_vocab()
    # Sort the vocab dictionary by ID. Result is tokens where the index equals the ID
    token_list = list(dict(sorted(token_list.items(), key=lambda item: item[1])).keys()) # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    # Make sure there are no duplicates in the ids_to_keep
    ids_to_keep = list(set(ids_to_keep))
    # Remove the ids_to_keep so they cannot be collapsed. Sorted and reverse so that we can just pop.
    for idx in sorted(ids_to_keep, reverse=True):
        token_list.pop(idx)
    # Turn the tokens into IDs
    id_list = tokenizer.convert_tokens_to_ids(token_list)
    # Get the embeddings of all tokens
    embeddings = original_embedding_layer(torch.tensor(id_list)).detach().numpy()

    # Euclidian distance search if wanted. Currently not used.
    euc_index = faiss.IndexFlatL2(embeddings.shape[1])
    euc_index.add(embeddings)

    # Need to normalize embeddings first for cosine similarity with faiss
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector
    cos_index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    cos_index.add(normalized_embeddings)

    # Search for the distances between vectors
    start_time = time.time()
    print("Starting calculating distances")
    # Faiss search
    D, I = cos_index.search(normalized_embeddings, k=len(normalized_embeddings)//search_reduction_factor) # this algo scales with k as klogk, so it works to check all embeddings.
    print(f"Time taken: {time.time() - start_time}")
    # Remove the first element, which is itself
    D = D[:, 1:] 
    I = I[:, 1:]

    # Create a list of all the pairs and the distance between them
    dists = []
    for i in range(len(I)-1, -1, -1):
        for k in range(len(I[i])):
            dists.append((D[i][k], i, I[i][k]))
    # Sort the list for greedy collapse
    dists.sort()

    # Prepare data structures needed for mapping and collapse
    token_to_new_id = {}
    id_to_new_id = {}
    removed = set()
    no_collapse = set()
    i = -1
    # Loop until the the number of mappings reaches the collapse amount
    while len(token_to_new_id) < len(token_list) // collapse_factor:
        i += 1
        # If this is true, we have run out of pairs to look at
        if i >= len(dists):
            raise Exception("Not enough distances calculated to collapse. Try increasing k or decreasing the collapse factor")
        # If the word-to-collapse has already been collapsed or collapsed into, or the word-to-collapse-into has already collapsed
        if dists[i][1] in removed or dists[i][1] in no_collapse or dists[i][2] in removed:
            continue
        # Add the collapse to the mappings
        token_to_new_id[token_list[dists[i][1]]] = id_list[dists[i][2]]
        id_to_new_id[id_list[dists[i][1]]] = id_list[dists[i][2]]
        # Add them to the sets to prevent collapsing into ineligable token
        removed.add(dists[i][1])
        no_collapse.add(dists[i][2])
        
    # Save the mappings
    try:
        with open("token_to_new_id.pkl", "wb") as file:
            pickle.dump(token_to_new_id, file)
        with open("id_to_new_id.pkl", "wb") as file:
            pickle.dump(id_to_new_id, file)
    except Exception as e:
        raise Exception(f"Failed to save mappings. This is required before running create_dataset(): {e}")
    
    return token_to_new_id, id_to_new_id


def create_tokenizer(token_to_new_id, base_dir, new_dir, delete_existing=False):
    """
    Create a tokenizer using the collapsed mappings. Deprecated.
    
    Args:
        token_to_new_id: The collapsed mappings from token to ID
        base_dir: The original tokenizer's directory
        new_dir: The new tokenizer's directory
        delete_existing: Whether or not to delete an existing directory at new_dir
    
    Returns:
        A tuple of the new tokenizer, the new vocabulary size, and the original vocabulary size
    """
    # Create a copy of the tokenizer files
    try:
        create_dir(new_dir, delete_existing)
        copytree(base_dir, new_dir)
    except Exception as e:
        raise Exception(f"Couldn't copy the tokenizer. Make sure base_dir contains the tokenizer data from HuggingFace: {e}")
    
    # Edit the vocab file
    with open(f"{new_dir}/vocab.json", "r", encoding='utf-8') as file:
        vocab = json.load(file)
    # Save the original size of the vocab
    original_size = len(set(vocab.values()))
    for word in token_to_new_id:
        vocab[word] = token_to_new_id[word]
    with open(f"{new_dir}/vocab.json", "w") as file:
        json.dump(vocab, file, ensure_ascii=False)
        
    # Edit tokenizer
    with open(f"{new_dir}/tokenizer.json", "r", encoding='utf-8') as file:
        tokenizer_info = json.load(file)
    for word in token_to_new_id:
        tokenizer_info['model']['vocab'][word] = token_to_new_id[word]
    with open(f"{new_dir}/tokenizer.json", "w") as file:
        json.dump(tokenizer_info, file, ensure_ascii=False)
        
    # Load new tokenizer
    new_tokenizer = GPT2TokenizerFast.from_pretrained(new_dir)
    
    # Delete the dir
    delete_dir(new_dir)
        
    return new_tokenizer, len(set(vocab.values())), original_size


def create_dataset(dataset, tokenizer, id_to_new_id, id_to_fixed_idx=None, og_ds_dir="original_dataset.pkl", new_ds_dir="new_dataset.pkl", amount=1000):
    """
    Create a dataset of IDs with collapsed words mapped to their counterparts.
    
    Args:
        dataset: The original dataset you are reducing
        tokenizer: The original tokenizer used by the original model.
        id_to_new_id: The collapsed mapping of ID to ID
        id_to_fixed_idx: The mapping from the collapsed mapping to the re-numbered IDs. Should be default for train, and validation should be provided with the returned id_to_fixed_idx. 
        og_ds_dir: The directory to save the IDs for the original dataset
        new_ds_dir: The directory to save the IDs for the collapsed dataset
        amount: The amount to do
    
    Returns:
        A tuple of the new dataset and the `id to fixed id` mappings, to be used when creating the validation dataset
    """
    # Creating both the collapsed dataset, but also saving the IDs for the original dataset
    new_dataset = []
    original_dataset = []
    
    # If doing the train set, initialize id_to_fixed_idx. Otherwise, use the one provided for the validation set.
    if id_to_fixed_idx is None:
        id_to_fixed_idx = {}
        idx = 2 # Saving first two indices for special tokens
    else:
        idx = len(id_to_fixed_idx) + 2
    
    # Loop through the samples
    for i in tqdm(range(amount)):
        # Extract the text and encode it into IDs
        text = dataset[i]['text']
        tokenized_text = tokenizer.encode(text)
        # Save the original dataset tokens
        original_dataset.append(deepcopy(tokenized_text))
        # Loop through the tokens to make the collapsed dataset
        for j in range(len(tokenized_text)):
            id = tokenized_text[j]
            # Need to swap due to collapse
            if id in id_to_new_id:
                id = id_to_new_id[id]
            # If the id doesn't have a reduced index (for renumbering), make it one
            if id not in id_to_fixed_idx:
                id_to_fixed_idx[id] = idx
                idx += 1
            # Replace the token with the collapsed, fixed ID token
            tokenized_text[j] = id_to_fixed_idx[id]
        new_dataset.append(tokenized_text)
    
    # Save both datasets
    try:
        with open(new_ds_dir, "wb") as file:
            pickle.dump(new_dataset, file)
        with open(og_ds_dir, "wb") as file:
            pickle.dump(original_dataset, file)
    except Exception as e:
        raise Exception(f"Failed to save datasets. This is required for experimentation: {e}")
        
    return new_dataset, id_to_fixed_idx


def fix_tokenizer_indexing(token_to_new_id, base_dir, new_dir, delete_existing=False):
    """
    Take an existing tokenizer and fix the numbering to match the new vocabulary size. Deprecated.
    
    Args:
        token_to_new_id: The collapsed mappings from token to ID
        base_dir: The original tokenizer directory
        new_dir: The new tokenizer directory
        delete_existing: Whether or not to delete an existing directory at new_dir
    
    Returns:
        The new tokenizer
    """
    # Create the tokenizer files by copying from the original ones
    try:
        create_dir(new_dir, delete_existing)
        copytree(base_dir, new_dir)
    except Exception as e:
        raise Exception(f"Couldn't copy the tokenizer. Make sure base_dir contains the tokenizer data from HuggingFace: {e}")
    
    # Create new vocab indexing
    with open(f"{new_dir}/vocab.json", "r", encoding='utf-8') as file:
        vocab = json.load(file)
    i = 0
    new_vocab = {}
    for word in vocab:
        if word in token_to_new_id:
            continue
        new_vocab[word] = i
        i += 1
        
    # Edit the vocab file
    with open(f"{new_dir}/vocab.json", "w", encoding='utf-8') as file:
        json.dump(new_vocab, file, ensure_ascii=False)
    # Edit the tokenizer file
    with open(f"{new_dir}/tokenizer.json", "r", encoding='utf-8') as file:
        tokenizer_info = json.load(file)
    tokenizer_info['model']['vocab'] = new_vocab
    # Note, that this causes issues, which is why this stuff is deprecated:
    tokenizer_info['model']['merges'] = []
    with open(f"{new_dir}/tokenizer.json", "w", encoding='utf-8') as file:
        json.dump(tokenizer_info, file, ensure_ascii=False)
    
    # Load new tokenizer
    new_tokenizer = GPT2TokenizerFast.from_pretrained(new_dir)
    
    return new_tokenizer


if __name__ == "__main__":
    # Load the dataset
    dataset = load_dataset('roneneldan/TinyStories')
    dataset_train = dataset['train']
    dataset_val = dataset['validation']

    # Model and tokenizer for tinystories dataset
    og_model = AutoModelForCausalLM.from_pretrained('roneneldan/TinyStories-33M')
    og_tokenizer = AutoTokenizer.from_pretrained("roneneldan/TinyStories-33M")
    bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased") # https://huggingface.co/docs/transformers/en/main_classes/tokenizer

    # Extract the embedding layer
    embedding_layer = og_model.transformer.wte

    COLLAPSE_FACTOR = 1/0.5 # Have to do the inverse for this code
    
    # token_to_new_id, id_to_new_id = get_collapsed_words(COLLAPSE_FACTOR, og_tokenizer, embedding_layer, use_saved=True, ids_to_keep=list(range(256)) + [-1])
    
    # collapsed_tokenizer, _, _ = create_tokenizer(token_to_new_id, "base_tokenizer", "custom_tokenizer", delete_existing=True)
    
    # perc_collapse = round(1/COLLAPSE_FACTOR, 2)
    # collapsed_dataset_train, id_to_fixed_idx = create_dataset(dataset_train, og_tokenizer, id_to_new_id, og_ds_dir=f"original_dataset.pkl", new_ds_dir=f"new_dataset_{perc_collapse}.pkl")
    # collapsed_dataset_test = create_dataset(dataset_val, og_tokenizer, id_to_new_id, id_to_fixed_idx, og_ds_dir=f"original_dataset_val.pkl", new_ds_dir=f"new_dataset_val_{perc_collapse}.pkl")

    # new_tokenizer = fix_tokenizer_indexing(token_to_new_id, "base_tokenizer", "custom_tokenizer", delete_existing=True)