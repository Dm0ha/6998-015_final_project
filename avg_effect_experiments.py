import time
from transformers import BertConfig
from utils import get_vocab_size, create_base_tokenizer
import pickle
from matplotlib import pyplot as plt
from transformers import BertForMaskedLM
import os
import numpy as np
from scipy import stats
from bert import MAX_LEN, pad_and_mask, add_mlm_masking, train, get_p_diff


"""
----------------------------------------------------------------------------------------
This script is used for conducting the experiment on the average error of the technique.
----------------------------------------------------------------------------------------
"""

def run_avg_effect(new_embed_size, original_embed_size, new_vocab_size, original_vocab_size, 
                   dir_loss, dir_model, collapse_factor, 
                   original_dataset_ids, original_dataset_mask, original_dataset_labels,
                   original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels,
                   new_dataset_ids, new_dataset_mask, new_dataset_labels,
                   new_dataset_val_ids, new_dataset_val_mask, new_dataset_val_labels,
                   i, epochs=75):
    """
    Run an experiment measuring the error of the technique for a particular iteration.
    
    Args:
        new_embed_size: Embedding dimensionality of the "new" model
        original_embed_size: Embedding dimensionality of the "old" model
        new_vocab_size: Vocabulary size of the "new" model
        original_vocab_size: Vocabulary size of the "old" model
        dir_loss: Directory to store the losses of the models trained
        dir_model: Directory to store the models trained
        collapse_factor: Collapse factor in the form 1/0.6 for 60% reduction
        
        original_dataset_ids: Dataset of IDs for the non-collapsed training data
        original_dataset_mask: Padding mask for the non-collapsed training data
        original_dataset_labels: MLM labels for the non-collapsed training data
        
        original_dataset_val_ids: Dataset of IDs for the non-collapsed validation data
        original_dataset_val_mask: Padding mask for the non-collapsed validation data
        original_dataset_val_labels: MLM labels for the non-collapsed validation data
        
        new_dataset_ids: Dataset of IDs for the collapsed training data
        new_dataset_mask: Padding mask for the collapsed training data
        new_dataset_labels: MLM labels for the collapsed training data
        
        new_dataset_val_ids: Dataset of IDs for the collapsed validation data
        new_dataset_val_mask: Padding mask for the collapsed validation data
        new_dataset_val_labels: MLM labels for the collapsed validation data
        
        i: Iteration index representing what trial this experiment is for, used in the file name
        epochs: The number of epochs to run
    """
    
    # Prepare the configs for each of the 4 models
    
    # Original vocab, original architecture
    bert_config_1 = BertConfig(
        vocab_size=original_vocab_size,
        hidden_size=original_embed_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        max_position_embeddings=MAX_LEN,
    )

    # New vocab, original architecture
    bert_config_2 = BertConfig(
        vocab_size=new_vocab_size,
        hidden_size=original_embed_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        max_position_embeddings=MAX_LEN,
    )

    # Original vocab, new architecture
    bert_config_3 = BertConfig(
        vocab_size=original_vocab_size,
        hidden_size=new_embed_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        max_position_embeddings=MAX_LEN,
    )

    # New vocab, new architecture
    bert_config_4 = BertConfig(
        vocab_size=new_vocab_size,
        hidden_size=new_embed_size,
        num_hidden_layers=1,
        num_attention_heads=1,
        intermediate_size=64,
        max_position_embeddings=MAX_LEN,
    )
    
    # For the original DATASET, only need to train once
    if not os.path.exists(f"{dir_loss}{collapse_factor}_BASE_odom.pkl"):
        # Create and train the old dataset, old model combination
        model = BertForMaskedLM(bert_config_1)
        _, old_d_old_m = train(original_dataset_ids, original_dataset_mask, original_dataset_labels,
                               original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels, model, epochs)
        # Save the model and loss
        with open(f"{dir_loss}{collapse_factor}_BASE_odom.pkl", "wb") as file:
            pickle.dump(old_d_old_m, file)
        model.save_pretrained(f"{dir_loss}{collapse_factor}_BASE_odom.pkl")
    else:
        print("Using saved model for old dataset")
        with open(f"{dir_loss}{collapse_factor}_BASE_odom.pkl", "rb") as file:
            old_d_old_m = pickle.load(file)

    # Create and train the new dataset, old model combination
    model = BertForMaskedLM(bert_config_2)
    _, new_d_old_m = train(new_dataset_ids, new_dataset_mask, new_dataset_labels,
                           new_dataset_val_ids, new_dataset_val_mask, new_dataset_val_labels, model, epochs)
    # Save the model and loss
    with open(f"{dir_loss}{collapse_factor}_{i}_ndom.pkl", "wb") as file:
        pickle.dump(new_d_old_m, file)
    model.save_pretrained(f"{dir_model}{collapse_factor}_{i}_ndom")
    
    # For the original DATASET, only need to train once
    if not os.path.exists(f"{dir_loss}{collapse_factor}_BASE_odnm.pkl"):
        # Create and train the old dataset, new model combination
        model = BertForMaskedLM(bert_config_3)
        _, old_d_new_m = train(original_dataset_ids, original_dataset_mask, original_dataset_labels,
                               original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels, model, epochs)
        # Save the model and loss
        with open(f"{dir_loss}{collapse_factor}_BASE_odnm.pkl", "wb") as file:
            pickle.dump(old_d_new_m, file)
        model.save_pretrained(f"{dir_loss}{collapse_factor}_BASE_odnm.pkl")
    else:
        print("Using saved model for old dataset")
        with open(f"{dir_loss}{collapse_factor}_BASE_odnm.pkl", "rb") as file:
            old_d_new_m = pickle.load(file)

    # Create and train the new dataset, new model combination
    model = BertForMaskedLM(bert_config_4)
    _, new_d_new_m = train(new_dataset_ids, new_dataset_mask, new_dataset_labels,
                           new_dataset_val_ids, new_dataset_val_mask, new_dataset_val_labels, model, epochs)
    # Save the model and loss
    with open(f"{dir_loss}{collapse_factor}_{i}_ndnm.pkl", "wb") as file:
        pickle.dump(new_d_new_m, file)
    model.save_pretrained(f"{dir_model}{collapse_factor}_{i}_ndnm")



if __name__ == "__main__":
    # Specify collapse factor and experiment directories
    collapse_factor = 2
    dir_loss = "experiment_data/avg_effects/losses/"
    dir_model = "experiment_data/avg_effects/models/"

    # Get both vocab sizes
    try:
        original_vocab_size = get_vocab_size("base_tokenizer")
    except Exception as e:
        print("Missing tokenizer, downloading the default TinyStories tokenizer.")
        create_base_tokenizer("roneneldan/TinyStories-33M")
        original_vocab_size = get_vocab_size("base_tokenizer")
    new_vocab_size = int(original_vocab_size // collapse_factor)

    # Load the datasets created in the collapse stage
    try:
        with open("new_dataset.pkl", "rb") as file:
            new_dataset_train = pickle.load(file)
        with open("original_dataset.pkl", "rb") as file:
            original_dataset_train = pickle.load(file)
        with open("new_dataset_val.pkl", "rb") as file:
            new_dataset_val = pickle.load(file)
        with open("original_dataset_val.pkl", "rb") as file:
            original_dataset_val = pickle.load(file)
    except Exception as e:
        raise Exception(f"Could not find the datasets. Make sure to run the collapse sequence first: {e}")

    # Prepare the datasets for the MLM task
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
        
    # Set the number of experiments
    EXPERIMENTS = 200

    # For each experiment, train the models for new loss values
    for i in range(EXPERIMENTS):
        # 32 to 128 is good because we want a significant performance difference, which it had in the embedding experiments.
        # undertrained, but should not matter for the purposes of this experiment
        run_avg_effect(128, 32, new_vocab_size, original_vocab_size, 
                       dir_loss, dir_model, collapse_factor, 
                       original_dataset_ids, original_dataset_mask, original_dataset_labels,
                       original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels,
                       new_dataset_ids, new_dataset_mask, new_dataset_labels,
                       new_dataset_val_ids, new_dataset_val_mask, new_dataset_val_labels,
                       i, epochs=10) 

    # ---- Plotting ----
    # Load the original dataset losses
    with open(f"{dir_loss}{collapse_factor}_BASE_odom.pkl", "rb") as file:
        odom = pickle.load(file)
    with open(f"{dir_loss}{collapse_factor}_BASE_odnm.pkl", "rb") as file:
        odnm = pickle.load(file)

    diffs = []
    n_ds = []
    o_ds = []
    for i in range(EXPERIMENTS):
        # Grab the new dataset losses for the particular experiment
        with open(f"{dir_loss}{collapse_factor}_{i}_ndom.pkl", "rb") as file:
            ndom = pickle.load(file)
        with open(f"{dir_loss}{collapse_factor}_{i}_ndnm.pkl", "rb") as file:
            ndnm = pickle.load(file)
        # Calculate the the difference between datasets for the percent difference between models
        diffs.append(abs(get_p_diff(ndom[-1], ndnm[-1]) - get_p_diff(odom[-1], odnm[-1])))
        # Store the percent difference between models of each dataset type
        n_ds.append(get_p_diff(ndom[-1], ndnm[-1]))
        o_ds.append(get_p_diff(odom[-1], odnm[-1]))

    # TOST Equivalence testing:
    # https://stackoverflow.com/questions/76069028/scipy-to-perform-a-t-test-for-equality
    n_ds = np.array(n_ds)
    o_ds = np.array(o_ds)
    delta = 0.4
    # Run each side's ttest and print it
    res1 = stats.ttest_ind(n_ds, o_ds - delta, alternative='greater')
    print(res1)
    res2 = stats.ttest_ind(n_ds, o_ds + delta, alternative='less')
    print(res2)

    # Sort the differences for percentile calculations
    diffs.sort()
    # Percentile calculations
    perc_90 = diffs[round(0.9 * EXPERIMENTS)]
    perc_95 = diffs[round(0.95 * EXPERIMENTS)]
    perc_99 = diffs[round(0.99 * EXPERIMENTS)]
    print("90th percentile:", perc_90)
    print("95th percentile:", perc_95)
    print("99th percentile:", perc_99)

    # Plot a CDF:
    # https://stackoverflow.com/questions/39728723/vertical-line-at-the-end-of-a-cdf-histogram-using-matplotlib
    n = np.arange(1,len(diffs)+1) / float(len(diffs))
    plt.plot(diffs, n)
    plt.ylim(0, 1)

    # Add lines for the percentiles
    plt.axvline(perc_90, ymax=0.9, color="black", linestyle="--", alpha=0.5, label="90th percentile")
    plt.text(perc_90 - 0.03, 0.1, round(perc_90, 2), ha='center', va='bottom')
    plt.axvline(perc_95, ymax=0.95, color="black", linestyle="-.", alpha=0.5, label="95th percentile")
    plt.text(perc_95 - 0.03, 0.1, round(perc_95, 2), ha='center', va='bottom')
    plt.axvline(perc_99, ymax=0.99, color="black", linestyle=":", alpha=0.5, label="99th percentile")
    plt.text(perc_99 - 0.03, 0.1, round(perc_99, 2), ha='center', va='bottom')

    # Add labels and display the graph
    plt.xlabel("Absolute Error in % Change Predictions (percentage points)")
    plt.ylabel("Cumulative Probability")
    plt.title("CDF of Error in % Change Predictions by Collapsed Dataset Models")
    plt.legend()
    plt.show()