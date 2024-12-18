import time
from transformers import BertConfig
from utils import get_vocab_size, num_params
import pickle
from matplotlib import pyplot as plt
from transformers import BertForMaskedLM
import os
import numpy as np

from bert import MAX_LEN, pad_and_mask, add_mlm_masking, train, get_p_diffs, get_p_diff


dir_loss = "experiment_data/collapse/losses/"
dir_model = "experiment_data/collapse/models/"

original_vocab_size = get_vocab_size("base_tokenizer")

# load original datasets
with open("original_dataset.pkl", "rb") as file:
    original_dataset_train = pickle.load(file)
with open("original_dataset_val.pkl", "rb") as file:
    original_dataset_val = pickle.load(file)

original_dataset_ids, original_dataset_mask = pad_and_mask(original_dataset_train)
original_dataset_ids, original_dataset_labels = add_mlm_masking(original_dataset_ids, original_vocab_size)
original_dataset_val_ids, original_dataset_val_mask = pad_and_mask(original_dataset_val)
original_dataset_val_ids, original_dataset_val_labels = add_mlm_masking(original_dataset_val_ids, original_vocab_size)


def run_embedding_test(collapse_factor, times, i, epochs=50):

    new_vocab_size = int(original_vocab_size * (1 - collapse_factor))

    with open(f"new_dataset_{collapse_factor}.pkl", "rb") as file:
        new_dataset_train = pickle.load(file)
    with open(f"new_dataset_val_{collapse_factor}.pkl", "rb") as file:
        new_dataset_val = pickle.load(file)

    new_dataset_ids, new_dataset_mask = pad_and_mask(new_dataset_train)
    new_dataset_ids, new_dataset_labels = add_mlm_masking(new_dataset_ids, new_vocab_size)
    new_dataset_val_ids, new_dataset_val_mask = pad_and_mask(new_dataset_val)
    new_dataset_val_ids, new_dataset_val_labels = add_mlm_masking(new_dataset_val_ids, new_vocab_size)

    bert_config_1 = BertConfig(
        vocab_size=original_vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=MAX_LEN,
    )

    bert_config_2 = BertConfig(
        vocab_size=new_vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=MAX_LEN,
    )

    bert_config_3 = BertConfig(
        vocab_size=original_vocab_size,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=MAX_LEN,
    )

    bert_config_4 = BertConfig(
        vocab_size=new_vocab_size,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=MAX_LEN,
    )

    original_params = num_params(BertForMaskedLM(bert_config_1))
    new_params = num_params(BertForMaskedLM(bert_config_2))
    print(f"CF-{collapse_factor} % change: {(new_params - original_params)/ original_params}")

    # For the original DATASET, only need to train once
    if not os.path.exists(f"{dir_loss}_odom.pkl"):
        model = BertForMaskedLM(bert_config_1)
        start_time = time.time()
        _, old_d_old_m = train(original_dataset_ids, original_dataset_mask, original_dataset_labels,
                               original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels, model, epochs)
        times["BASELINE"] = time.time() - start_time
        with open(f"{dir_loss}_odom.pkl", "wb") as file:
            pickle.dump(old_d_old_m, file)
        model.save_pretrained(f"{dir_model}_odom")
    else:
        print("Using saved model for old dataset")
        with open(f"{dir_loss}_odom.pkl", "rb") as file:
            old_d_old_m = pickle.load(file)

    model = BertForMaskedLM(bert_config_2)
    start_time = time.time()
    _, new_d_old_m = train(new_dataset_ids, new_dataset_mask, new_dataset_labels,
                            new_dataset_val_ids, new_dataset_val_mask, new_dataset_val_labels, model, epochs)
    times[collapse_factor] = time.time() - start_time
    with open(f"{dir_loss}{collapse_factor}_{i}_ndom.pkl", "wb") as file:
        pickle.dump(new_d_old_m, file)
    model.save_pretrained(f"{dir_model}{collapse_factor}_{i}_ndom")

    # For the original DATASET, only need to train once
    if not os.path.exists(f"{dir_loss}_odnm.pkl"):
        model = BertForMaskedLM(bert_config_3)
        _, old_d_new_m = train(original_dataset_ids, original_dataset_mask, original_dataset_labels,
                               original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels, model, epochs)
        with open(f"{dir_loss}_odnm.pkl", "wb") as file:
            pickle.dump(old_d_new_m, file)
        model.save_pretrained(f"{dir_model}_odnm")
    else:
        print("Using saved model for old dataset")
        with open(f"{dir_loss}_odnm.pkl", "rb") as file:
            old_d_new_m = pickle.load(file)

    model = BertForMaskedLM(bert_config_4)
    _, new_d_new_m = train(new_dataset_ids, new_dataset_mask, new_dataset_labels,
                           new_dataset_val_ids, new_dataset_val_mask, new_dataset_val_labels, model, epochs)
    with open(f"{dir_loss}{collapse_factor}_{i}_ndnm.pkl", "wb") as file:
        pickle.dump(new_d_new_m, file)
    model.save_pretrained(f"{dir_model}{collapse_factor}_{i}_ndnm")


collapse_factors = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
times = {}
for collapse_factor in collapse_factors:
    for i in range(10):
        run_embedding_test(collapse_factor, times, i, epochs=50)
print(times)

# Plotting
with open(f"{dir_loss}_odom.pkl", "rb") as file:
    odom = pickle.load(file)
with open(f"{dir_loss}_odnm.pkl", "rb") as file:
    odnm = pickle.load(file)

ys = []

for collapse_factor in collapse_factors:
    itr_ys = []
    for i in range(10):
        with open(f"{dir_loss}{collapse_factor}_{i}_ndom.pkl", "rb") as file:
            ndom = pickle.load(file)
        with open(f"{dir_loss}{collapse_factor}_{i}_ndnm.pkl", "rb") as file:
            ndnm = pickle.load(file)
        itr_ys.append(abs(get_p_diff(odom[-1], odnm[-1]) - get_p_diff(ndom[-1], ndnm[-1])))
    ys.append(sum(itr_ys) / len(itr_ys))

# https://www.statology.org/matplotlib-trendline/
z = np.polyfit(collapse_factors, ys, 1)
p = np.poly1d(z)

plt.plot(collapse_factors, ys)
plt.plot(collapse_factors, p(collapse_factors), linestyle="--", color="black", alpha=0.5)
plt.xlabel("Collapse Factor")
plt.ylabel("Average Absolute Error in % Change Predictions (percentage points)")
plt.title("Error in % Change Predictions by Collapsed Dataset Models Varying Collapse Factor")
plt.show()