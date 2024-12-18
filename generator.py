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

# https://stackoverflow.com/questions/69780823/tokenizers-change-vocabulary-entry

        
def get_bert_tokenizer_vocab():
    token_set = set()
    for i in tqdm(range(len(dataset_train))):
        story = dataset_train[i]['text']
        tokenized_story = bert_tokenizer.tokenize(story, return_tensors='pt', padding=False)
        token_set.update(tokenized_story)
    token_set = token_set - set(bert_tokenizer.all_special_tokens) # For UNK in particular
    with open("token_list.pkl", "wb") as file:
        pickle.dump(list(token_set), file)


def build_starting_bert_tokenizer():
    # Load the token set
    with open("token_list.pkl", "rb") as file:
        token_list = pickle.load(file)
        
    # prepare the tokens for saving into tokenizer files
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    token_list = special_tokens + token_list
    vocab = {}
    for i in range(len(token_list)):
        vocab[token_list[i]] = i
    
    # Make the initial directory
    create_dir("bert_filtered_tokenizer", delete_existing=True)
    copytree("bert_tokenizer", "bert_filtered_tokenizer")
    
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
    with open("bert_filtered_tokenizer/tokenizer.json", "w") as file:
        json.dump(tokenizer_info, file)
        

def get_collapsed_words(collapse_factor, tokenizer, original_embedding_layer, use_saved=False, ids_to_keep=[]):
    # Use the saved versions if they exist
    if use_saved and os.path.exists("token_to_new_id.pkl") and os.path.exists("id_to_new_id.pkl"):
        with open("token_to_new_id.pkl", "rb") as file:
            token_to_new_id = pickle.load(file)
        with open("id_to_new_id.pkl", "rb") as file:
            id_to_new_id = pickle.load(file)
        return token_to_new_id, id_to_new_id
    
    token_list = tokenizer.get_vocab()
    token_list = list(dict(sorted(token_list.items(), key=lambda item: item[1])).keys()) # https://stackoverflow.com/questions/613183/how-do-i-sort-a-dictionary-by-value
    ids_to_keep = list(set(ids_to_keep))
    for idx in sorted(ids_to_keep, reverse=True):
        token_list.pop(idx)
    id_list = tokenizer.convert_tokens_to_ids(token_list)
    embeddings = original_embedding_layer(torch.tensor(id_list)).detach().numpy()
    
    # print(token_set & set(tokenizer.all_special_ids))

    euc_index = faiss.IndexFlatL2(embeddings.shape[1])
    euc_index.add(embeddings)

    # Need to normalize embeddings first for cosine similarity with faiss
    normalized_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True) # https://stackoverflow.com/questions/21030391/how-to-normalize-a-numpy-array-to-a-unit-vector
    cos_index = faiss.IndexFlatIP(normalized_embeddings.shape[1])
    cos_index.add(normalized_embeddings)

    # Search for distances
    start_time = time.time()
    print("Starting calculating distances")
    D, I = cos_index.search(normalized_embeddings, k=len(normalized_embeddings)//25) # this algo scales with k as klogk, so it works to check all embeddings.
    print(f"Time taken: {time.time() - start_time}")
    # Remove the first element, which is itself
    D = D[:, 1:] 
    I = I[:, 1:]

    dists = []
    for i in range(len(I)-1, -1, -1):
        for k in range(len(I[i])):
            dists.append((D[i][k], i, I[i][k]))
    dists.sort()

    token_to_new_id = {}
    id_to_new_id = {}
    removed = set()
    no_collapse = set()
    i = -1
    while len(token_to_new_id) < len(token_list) // collapse_factor:
        i += 1
        if i >= len(dists):
            raise Exception("Not enough distances calculated to collapse. Try increasing k or decreasing the collapse factor")
        # If word-to-collapse has already been collapsed or collapsed into, or the word-to-collapse-into has already collapsed
        if dists[i][1] in removed or dists[i][1] in no_collapse or dists[i][2] in removed:
            continue
        token_to_new_id[token_list[dists[i][1]]] = id_list[dists[i][2]]
        id_to_new_id[id_list[dists[i][1]]] = id_list[dists[i][2]]
        removed.add(dists[i][1])
        no_collapse.add(dists[i][2])
        
    # Save the mappings
    with open("token_to_new_id.pkl", "wb") as file:
        pickle.dump(token_to_new_id, file)
    with open("id_to_new_id.pkl", "wb") as file:
        pickle.dump(id_to_new_id, file)
    
    return token_to_new_id, id_to_new_id

# token_to_new_id, id_to_new_id = get_collapsed_words(COLLAPSE_FACTOR, og_tokenizer, embedding_layer, use_saved=True, ids_to_keep=list(range(256)) + [-1])


# Create new tokenizer
def create_tokenizer(token_to_new_id, base_dir, new_dir, delete_existing=False):
    # Create a copy of the tokenizer files
    create_dir(new_dir, delete_existing)
    copytree(base_dir, new_dir)
    
    # Edit the vocab file
    with open(f"{new_dir}/vocab.json", "r", encoding='utf-8') as file:
        vocab = json.load(file)
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
    
# collapsed_tokenizer, _, _ = create_tokenizer(token_to_new_id, "base_tokenizer", "custom_tokenizer", delete_existing=True)

def create_dataset(dataset, tokenizer, id_to_new_id, id_to_fixed_idx=None, og_ds_dir="original_dataset.pkl", new_ds_dir="new_dataset.pkl", amount=1000):
    new_dataset = []
    original_dataset = []
    if id_to_fixed_idx is None:
        id_to_fixed_idx = {}
        idx = 2 # Saving first two indices for special tokens
    else:
        idx = len(id_to_fixed_idx) + 2
    for i in tqdm(range(amount)):
        text = dataset[i]['text']
        tokenized_text = tokenizer.encode(text)
        original_dataset.append(deepcopy(tokenized_text))
        for j in range(len(tokenized_text)):
            id = tokenized_text[j]
            # Need to swap due to collapse
            if id in id_to_new_id:
                id = id_to_new_id[id]
            # If the id doesn't have a reduced index, make it one
            if id not in id_to_fixed_idx:
                id_to_fixed_idx[id] = idx
                idx += 1
            tokenized_text[j] = id_to_fixed_idx[id]
        new_dataset.append(tokenized_text)
    
    with open(new_ds_dir, "wb") as file:
        pickle.dump(new_dataset, file)
    with open(og_ds_dir, "wb") as file:
        pickle.dump(original_dataset, file)
        
    return new_dataset, id_to_fixed_idx


def fix_tokenizer_indexing(token_to_new_id, base_dir, new_dir, delete_existing=False):
    create_dir(new_dir, delete_existing)
    copytree(base_dir, new_dir)
    
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
        
    # Edit the files
    with open(f"{new_dir}/vocab.json", "w", encoding='utf-8') as file:
        json.dump(new_vocab, file, ensure_ascii=False)
    with open(f"{new_dir}/tokenizer.json", "r", encoding='utf-8') as file:
        tokenizer_info = json.load(file)
    tokenizer_info['model']['vocab'] = new_vocab
    tokenizer_info['model']['merges'] = []
    with open(f"{new_dir}/tokenizer.json", "w", encoding='utf-8') as file:
        json.dump(tokenizer_info, file, ensure_ascii=False)
    
    # Load new tokenizer
    new_tokenizer = GPT2TokenizerFast.from_pretrained(new_dir)
    
    return new_tokenizer
    
# perc_collapse = round(1/COLLAPSE_FACTOR, 2)
# collapsed_dataset_train, id_to_fixed_idx = create_dataset(dataset_train, og_tokenizer, id_to_new_id, og_ds_dir=f"original_dataset.pkl", new_ds_dir=f"new_dataset_{perc_collapse}.pkl")
# collapsed_dataset_test = create_dataset(dataset_val, og_tokenizer, id_to_new_id, id_to_fixed_idx, og_ds_dir=f"original_dataset_val.pkl", new_ds_dir=f"new_dataset_val_{perc_collapse}.pkl")

# new_tokenizer = fix_tokenizer_indexing(token_to_new_id, "base_tokenizer", "custom_tokenizer", delete_existing=True)