import os
import shutil
import json
from huggingface_hub import snapshot_download

"""
------------------------------------------------------------------------------------
This file contains a few utility functions that are used in other parts of the code.
------------------------------------------------------------------------------------
"""

# https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
def copytree(src, dst):
    """
    Create a directory to another directory
    
    Args:
        src: The source directory
        dst: The destination directory
    """
    # Go through each item in a directory
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        # If the item is a directory, recurse
        if os.path.isdir(s):
            shutil.copytree(s, d, False, None)
        # Otherwise, copy the file
        else:
            shutil.copy2(s, d)

# https://www.geeksforgeeks.org/create-a-directory-in-python/#
def create_dir(directory_name, delete_existing=False):
    """
    Creates a directory
    
    Args:
        directory_name: The directory to create
        delete_existing: Whether or not to delete the directory/file with the same name
    """
    try:
        # Make the directory
        os.mkdir(directory_name)
    except FileExistsError:
        # If it exists, delete depending on the delete_existing parameter
        if delete_existing:
            delete_dir(directory_name)
            os.mkdir(directory_name)
        else:
            print(f"Directory '{directory_name}' already exists.")
    except PermissionError:
        # No access
        print(f"Permission denied: Unable to create '{directory_name}'.")
    except Exception as e:
        # General error
        print(f"An error occurred: {e}")
        
def delete_dir(directory_name):
    """
    Deletes a directory
    
    Args:
        directory_name: The directory to create
    """
    try:
        # Using shutil to delete the directory
        shutil.rmtree(directory_name) # https://stackoverflow.com/questions/6996603/how-can-i-delete-a-file-or-folder-in-python
    except Exception as e:
        # General error
        print(f"Couldn't delete dir: {e}")
        
def get_vocab_size(tokenizer_dir):
    """
    Gets the vocabulary size from a tokenizer directory
    
    Args:
        tokenizer_dir: The directory of the tokenizer
    """
    # Open the tokenizer vocab
    with open(f"{tokenizer_dir}/vocab.json", "r", encoding='utf-8') as file:
        vocab = json.load(file)
    # Calculate the number of tokens
    return len(set(vocab.values()))

def num_params(model):
    """
    Gets the number of params in a model
    
    Args:
        model: The model, which should be built with PyTorch
    """
    total = 0
    # Loop through parameters
    for p in model.parameters():
        # Add the number of elements in the tensor
        total += p.numel()
    return total


def create_base_tokenizer(id):
    """
    Downloads and formats the base tokenizer files, used primarily for word counts.
    
    Args:
        id: The hugging face ID, e.g. "roneneldan/TinyStories-33M"
    """
    snapshot_download(repo_id=id, cache_dir="./base_tokenizer", local_dir_use_symlinks=False)
    # https://stackoverflow.com/questions/1724693/find-a-file-in-python
    for root, _, files in os.walk("base_tokenizer"):
        # Find each important file and copy it to the top of the directory
        for name in ["config.json", "merges.txt", "pytorch_model.bin", "special_tokens_map.json", "tokenizer_config.json", "tokenizer.json", "vocab.json"]:
            if name in files:
                shutil.copy(os.path.join(root, name), "base_tokenizer")