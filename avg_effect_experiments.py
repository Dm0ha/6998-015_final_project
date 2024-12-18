import time
from transformers import BertConfig
from utils import get_vocab_size
import pickle
from matplotlib import pyplot as plt
from transformers import BertForMaskedLM
import os
import numpy as np
from scipy import stats

from bert import MAX_LEN, pad_and_mask, add_mlm_masking, train, get_p_diffs, get_p_diff


COLLAPSE_FACTOR = 2
dir_loss = "experiment_data/avg_effects/losses/"
dir_model = "experiment_data/avg_effects/models/"

original_vocab_size = get_vocab_size("base_tokenizer")
new_vocab_size = int(original_vocab_size // COLLAPSE_FACTOR)

with open("new_dataset.pkl", "rb") as file:
    new_dataset_train = pickle.load(file)
with open("original_dataset.pkl", "rb") as file:
    original_dataset_train = pickle.load(file)
with open("new_dataset_val.pkl", "rb") as file:
    new_dataset_val = pickle.load(file)
with open("original_dataset_val.pkl", "rb") as file:
    original_dataset_val = pickle.load(file)

print(new_dataset_train[99])
print(original_dataset_train[99])

start = time.time()
print("Creating mlm dataset")
new_dataset_ids, new_dataset_mask = pad_and_mask(new_dataset_train)
new_dataset_ids, new_dataset_labels = add_mlm_masking(new_dataset_ids, new_vocab_size)
original_dataset_ids, original_dataset_mask = pad_and_mask(original_dataset_train)
original_dataset_ids, original_dataset_labels = add_mlm_masking(original_dataset_ids, original_vocab_size)

new_dataset_val_ids, new_dataset_val_mask = pad_and_mask(new_dataset_val)
new_dataset_val_ids, new_dataset_val_labels = add_mlm_masking(new_dataset_val_ids, new_vocab_size)
original_dataset_val_ids, original_dataset_val_mask = pad_and_mask(original_dataset_val)
original_dataset_val_ids, original_dataset_val_labels = add_mlm_masking(original_dataset_val_ids, original_vocab_size)
print(f"Time to craete dataset: {time.time() - start} sec")


def run_avg_effect(new_embed_size, original_embed_size, i, epochs=75):

    bert_config_1 = BertConfig(
        vocab_size=original_vocab_size,
        hidden_size=original_embed_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        max_position_embeddings=MAX_LEN,
    )

    bert_config_2 = BertConfig(
        vocab_size=new_vocab_size,
        hidden_size=original_embed_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        max_position_embeddings=MAX_LEN,
    )

    bert_config_3 = BertConfig(
        vocab_size=original_vocab_size,
        hidden_size=new_embed_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        max_position_embeddings=MAX_LEN,
    )

    bert_config_4 = BertConfig(
        vocab_size=new_vocab_size,
        hidden_size=new_embed_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        max_position_embeddings=MAX_LEN,
    )
    
    # For the original DATASET, only need to train once
    if not os.path.exists(f"{dir_loss}{COLLAPSE_FACTOR}_BASE_odom.pkl"):
        model = BertForMaskedLM(bert_config_1)
        _, old_d_old_m = train(original_dataset_ids, original_dataset_mask, original_dataset_labels,
                               original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels, model, epochs)
        with open(f"{dir_loss}{COLLAPSE_FACTOR}_BASE_odom.pkl", "wb") as file:
            pickle.dump(old_d_old_m, file)
        model.save_pretrained(f"{dir_loss}{COLLAPSE_FACTOR}_BASE_odom.pkl")
    else:
        print("Using saved model for old dataset")
        with open(f"{dir_loss}{COLLAPSE_FACTOR}_BASE_odom.pkl", "rb") as file:
            old_d_old_m = pickle.load(file)

    model = BertForMaskedLM(bert_config_2)
    _, new_d_old_m = train(new_dataset_ids, new_dataset_mask, new_dataset_labels,
                           new_dataset_val_ids, new_dataset_val_mask, new_dataset_val_labels, model, epochs)
    with open(f"{dir_loss}{COLLAPSE_FACTOR}_{i}_ndom.pkl", "wb") as file:
        pickle.dump(new_d_old_m, file)
    model.save_pretrained(f"{dir_model}{COLLAPSE_FACTOR}_{i}_ndom")
    
    # For the original DATASET, only need to train once
    if not os.path.exists(f"{dir_loss}{COLLAPSE_FACTOR}_BASE_odnm.pkl"):
        model = BertForMaskedLM(bert_config_3)
        _, old_d_new_m = train(original_dataset_ids, original_dataset_mask, original_dataset_labels,
                               original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels, model, epochs)
        with open(f"{dir_loss}{COLLAPSE_FACTOR}_BASE_odnm.pkl", "wb") as file:
            pickle.dump(old_d_new_m, file)
        model.save_pretrained(f"{dir_loss}{COLLAPSE_FACTOR}_BASE_odnm.pkl")
    else:
        print("Using saved model for old dataset")
        with open(f"{dir_loss}{COLLAPSE_FACTOR}_BASE_odnm.pkl", "rb") as file:
            old_d_new_m = pickle.load(file)

    model = BertForMaskedLM(bert_config_4)
    _, new_d_new_m = train(new_dataset_ids, new_dataset_mask, new_dataset_labels,
                           new_dataset_val_ids, new_dataset_val_mask, new_dataset_val_labels, model, epochs)
    with open(f"{dir_loss}{COLLAPSE_FACTOR}_{i}_ndnm.pkl", "wb") as file:
        pickle.dump(new_d_new_m, file)
    model.save_pretrained(f"{dir_model}{COLLAPSE_FACTOR}_{i}_ndnm")
    

EXPERIMENTS = 200

for i in range(EXPERIMENTS):
    # 32 to 128 is good because we want a significant performance difference, which it had in the embedding experiments.
    run_avg_effect(128, 32, i, epochs=10) # undertrained, but should not matter for the purposes of this experiment

# Plotting
with open(f"{dir_loss}{COLLAPSE_FACTOR}_BASE_odom.pkl", "rb") as file:
    odom = pickle.load(file)
with open(f"{dir_loss}{COLLAPSE_FACTOR}_BASE_odnm.pkl", "rb") as file:
    odnm = pickle.load(file)

diffs = []
n_ds = []
o_ds = []
for i in range(EXPERIMENTS):
    with open(f"{dir_loss}{COLLAPSE_FACTOR}_{i}_ndom.pkl", "rb") as file:
        ndom = pickle.load(file)
    with open(f"{dir_loss}{COLLAPSE_FACTOR}_{i}_ndnm.pkl", "rb") as file:
        ndnm = pickle.load(file)
    diffs.append(abs(get_p_diff(ndom[-1], ndnm[-1]) - get_p_diff(odom[-1], odnm[-1])))
    n_ds.append(get_p_diff(ndom[-1], ndnm[-1]))
    o_ds.append(get_p_diff(odom[-1], odnm[-1]))

# TOST Equivalence testing:
# https://stackoverflow.com/questions/76069028/scipy-to-perform-a-t-test-for-equality
n_ds = np.array(n_ds)
o_ds = np.array(o_ds)
delta = 0.4
res1 = stats.ttest_ind(n_ds, o_ds - delta, alternative='greater')
print(res1)
res2 = stats.ttest_ind(n_ds, o_ds + delta, alternative='less')
print(res2)

diffs.sort()
perc_90 = diffs[round(0.9 * EXPERIMENTS)]
perc_95 = diffs[round(0.95 * EXPERIMENTS)]
perc_99 = diffs[round(0.99 * EXPERIMENTS)]
print("90th percentile:", perc_90)
print("95th percentile:", perc_95)
print("99th percentile:", perc_99)
# plt.hist(diffs, bins=200, density=True, cumulative=True, histtype='step')
# plt.show()

# https://stackoverflow.com/questions/39728723/vertical-line-at-the-end-of-a-cdf-histogram-using-matplotlib
n = np.arange(1,len(diffs)+1) / float(len(diffs))
plt.plot(diffs, n)
plt.ylim(0, 1)

plt.axvline(perc_90, ymax=0.9, color="black", linestyle="--", alpha=0.5, label="90th percentile")
plt.text(perc_90 - 0.03, 0.1, round(perc_90, 2), ha='center', va='bottom')
plt.axvline(perc_95, ymax=0.95, color="black", linestyle="-.", alpha=0.5, label="95th percentile")
plt.text(perc_95 - 0.03, 0.1, round(perc_95, 2), ha='center', va='bottom')
plt.axvline(perc_99, ymax=0.99, color="black", linestyle=":", alpha=0.5, label="99th percentile")
plt.text(perc_99 - 0.03, 0.1, round(perc_99, 2), ha='center', va='bottom')

plt.xlabel("Absolute Error in % Change Predictions (percentage points)")
plt.ylabel("Cumulative Probability")
plt.title("CDF of Error in % Change Predictions by Collapsed Dataset Models")
plt.legend()
plt.show()