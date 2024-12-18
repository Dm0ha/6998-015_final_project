import os
import shutil
import json

# https://stackoverflow.com/questions/1868714/how-do-i-copy-an-entire-directory-of-files-into-an-existing-directory-using-pyth
def copytree(src, dst, symlinks=False, ignore=None):
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)

# https://www.geeksforgeeks.org/create-a-directory-in-python/#
def create_dir(directory_name, delete_existing=False):
    try:
        os.mkdir(directory_name)
    except FileExistsError:
        if delete_existing:
            delete_dir(directory_name)
            os.mkdir(directory_name)
        else:
            print(f"Directory '{directory_name}' already exists.")
    except PermissionError:
        print(f"Permission denied: Unable to create '{directory_name}'.")
    except Exception as e:
        print(f"An error occurred: {e}")
        
def delete_dir(directory_name):
    try:
        shutil.rmtree(directory_name) # https://stackoverflow.com/questions/6996603/how-can-i-delete-a-file-or-folder-in-python
    except Exception as e:
        print(f"Couldn't delete dir: {e}")
        
def get_vocab_size(tokenizer_dir):
    with open(f"{tokenizer_dir}/vocab.json", "r", encoding='utf-8') as file:
        vocab = json.load(file)
    return len(set(vocab.values()))

def num_params(model):
    total = 0
    for p in model.parameters():
        total += p.numel()
    return total