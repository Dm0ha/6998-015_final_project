import time
from transformers import BertConfig
from utils import get_vocab_size, num_params, create_base_tokenizer
import pickle
from matplotlib import pyplot as plt
from transformers import BertForMaskedLM
import os
import numpy as np
from bert import MAX_LEN, pad_and_mask, add_mlm_masking, train, get_p_diff

"""
--------------------------------------------------------------------------------------------
This script is used for conducting the experiment on how error changes with collapse factor.
--------------------------------------------------------------------------------------------
"""

def run_collapse_test(collapse_factor, i, original_vocab_size, dir_loss, dir_model, 
                      original_dataset_ids, original_dataset_mask, original_dataset_labels,
                      original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels,
                      times={}, epochs=50):
    """
    Run an experiment measuring error for a particular iteration of a collapse factor.
    
    Args:
        collapse_factor: Specifies which collapsed dataset to grab. In the form 0.6 for 60% reduction.
        i: Iteration index representing what trial this experiment is for, used in the loss file name
        original_vocab_size: Vocabulary size of the "old" model
        dir_loss: Directory to store the losses of the models trained
        dir_model: Directory to store the models trained
        
        original_dataset_ids: Dataset of IDs for the non-collapsed training data
        original_dataset_mask: Padding mask for the non-collapsed training data
        original_dataset_labels: MLM labels for the non-collapsed training data
        
        original_dataset_val_ids: Dataset of IDs for the non-collapsed validation data
        original_dataset_val_mask: Padding mask for the non-collapsed validation data
        original_dataset_val_labels: MLM labels for the non-collapsed validation data
        
        times: An optional parameter used to store the amount of time it takes to train the original model on both datasets
        epochs: The number of epochs to run
    """

    # Get the new vocab size
    new_vocab_size = int(original_vocab_size * (1 - collapse_factor))

    # Load the collapsed dataset
    try:
        with open(f"new_dataset_{collapse_factor}.pkl", "rb") as file:
            new_dataset_train = pickle.load(file)
        with open(f"new_dataset_val_{collapse_factor}.pkl", "rb") as file:
            new_dataset_val = pickle.load(file)
    except Exception as e:
        raise Exception(f"Could not find the collapsed datasets. Make sure to run the collapse sequence first: {e}")

    # Prepare the dataset for the MLM task
    new_dataset_ids, new_dataset_mask = pad_and_mask(new_dataset_train)
    new_dataset_ids, new_dataset_labels = add_mlm_masking(new_dataset_ids, new_vocab_size)
    new_dataset_val_ids, new_dataset_val_mask = pad_and_mask(new_dataset_val)
    new_dataset_val_ids, new_dataset_val_labels = add_mlm_masking(new_dataset_val_ids, new_vocab_size)

    # Prepare the configs for each of the 4 models
    
    # Original vocab, original architecture
    bert_config_1 = BertConfig(
        vocab_size=original_vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=MAX_LEN,
    )

    # New vocab, original architecture
    bert_config_2 = BertConfig(
        vocab_size=new_vocab_size,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=MAX_LEN,
    )

    # Original vocab, new architecture
    bert_config_3 = BertConfig(
        vocab_size=original_vocab_size,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=MAX_LEN,
    )

    # New vocab, new architecture
    bert_config_4 = BertConfig(
        vocab_size=new_vocab_size,
        hidden_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=MAX_LEN,
    )

    # Calulate the total parameters of the original model with both dataset vocabulary sizes
    original_params = num_params(BertForMaskedLM(bert_config_1))
    new_params = num_params(BertForMaskedLM(bert_config_2))
    print(f"CF-{collapse_factor} % change: {(new_params - original_params)/ original_params}")

    # For the original DATASET, only need to train once
    if not os.path.exists(f"{dir_loss}_odom.pkl"):
        # Create and train the old dataset, old model combination
        model = BertForMaskedLM(bert_config_1)
        # Measure the time it takes to train for this model
        start_time = time.time()
        _, old_d_old_m = train(original_dataset_ids, original_dataset_mask, original_dataset_labels,
                               original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels, model, epochs)
        times["BASELINE"] = time.time() - start_time
        # Save the model and loss
        with open(f"{dir_loss}_odom.pkl", "wb") as file:
            pickle.dump(old_d_old_m, file)
        model.save_pretrained(f"{dir_model}_odom")
    else:
        print("Using saved model for old dataset")
        with open(f"{dir_loss}_odom.pkl", "rb") as file:
            old_d_old_m = pickle.load(file)

    # Create and train the new dataset, old model combination
    model = BertForMaskedLM(bert_config_2)
    # Measure the time it takes to train for this model
    start_time = time.time()
    _, new_d_old_m = train(new_dataset_ids, new_dataset_mask, new_dataset_labels,
                            new_dataset_val_ids, new_dataset_val_mask, new_dataset_val_labels, model, epochs)
    times[collapse_factor] = time.time() - start_time
    # Save the model and loss
    with open(f"{dir_loss}{collapse_factor}_{i}_ndom.pkl", "wb") as file:
        pickle.dump(new_d_old_m, file)
    model.save_pretrained(f"{dir_model}{collapse_factor}_{i}_ndom")

    # For the original DATASET, only need to train once
    if not os.path.exists(f"{dir_loss}_odnm.pkl"):
        # Create and train the old dataset, new model combination
        model = BertForMaskedLM(bert_config_3)
        _, old_d_new_m = train(original_dataset_ids, original_dataset_mask, original_dataset_labels,
                               original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels, model, epochs)
        # Save the model and loss
        with open(f"{dir_loss}_odnm.pkl", "wb") as file:
            pickle.dump(old_d_new_m, file)
        model.save_pretrained(f"{dir_model}_odnm")
    else:
        print("Using saved model for old dataset")
        with open(f"{dir_loss}_odnm.pkl", "rb") as file:
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
    # Specify experiment directories
    dir_loss = "experiment_data/collapse/losses/"
    dir_model = "experiment_data/collapse/models/"

    # Get the original vocab size
    try:
        original_vocab_size = get_vocab_size("base_tokenizer")
    except Exception as e:
        print("Missing tokenizer, downloading the default TinyStories tokenizer.")
        create_base_tokenizer("roneneldan/TinyStories-33M")
        original_vocab_size = get_vocab_size("base_tokenizer")

    # Load the original dataset
    try:
        with open("original_dataset.pkl", "rb") as file:
            original_dataset_train = pickle.load(file)
        with open("original_dataset_val.pkl", "rb") as file:
            original_dataset_val = pickle.load(file)
    except Exception as e:
        raise Exception(f"Could not find the original datasets. Make sure to run the collapse sequence first: {e}")

    # Prepare the dataset for the MLM task
    original_dataset_ids, original_dataset_mask = pad_and_mask(original_dataset_train)
    original_dataset_ids, original_dataset_labels = add_mlm_masking(original_dataset_ids, original_vocab_size)
    original_dataset_val_ids, original_dataset_val_mask = pad_and_mask(original_dataset_val)
    original_dataset_val_ids, original_dataset_val_labels = add_mlm_masking(original_dataset_val_ids, original_vocab_size)

    # Set the number of trials to do for each collapse factor
    EXPERIMENTS = 10

    # Specify the collapse factors that we have created datasets for
    collapse_factors = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
    # Dict to save the time to save the train times on the original architecture
    times = {}
    for collapse_factor in collapse_factors:
        # Loop through each experiment and run each test
        for i in range(EXPERIMENTS):
            run_collapse_test(collapse_factor, i,original_vocab_size, dir_loss, dir_model, 
                              original_dataset_ids, original_dataset_mask, original_dataset_labels,
                              original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels,
                              times, epochs=50)
    print(times)

    # --- Plotting ---
    # Load the original dataset models
    with open(f"{dir_loss}_odom.pkl", "rb") as file:
        odom = pickle.load(file)
    with open(f"{dir_loss}_odnm.pkl", "rb") as file:
        odnm = pickle.load(file)

    ys = []
    for collapse_factor in collapse_factors:
        itr_ys = []
        for i in range(EXPERIMENTS):
            # Load the new dataset losses
            with open(f"{dir_loss}{collapse_factor}_{i}_ndom.pkl", "rb") as file:
                ndom = pickle.load(file)
            with open(f"{dir_loss}{collapse_factor}_{i}_ndnm.pkl", "rb") as file:
                ndnm = pickle.load(file)
            # Calculate the the difference between datasets for the percent difference between models
            itr_ys.append(abs(get_p_diff(odom[-1], odnm[-1]) - get_p_diff(ndom[-1], ndnm[-1])))
        # Save the average difference across the experiments
        ys.append(sum(itr_ys) / len(itr_ys))

    # Create a linear best fit:
    # https://www.statology.org/matplotlib-trendline/
    z = np.polyfit(collapse_factors, ys, 1)
    p = np.poly1d(z)

    # Plot the results
    plt.plot(collapse_factors, ys)
    plt.plot(collapse_factors, p(collapse_factors), linestyle="--", color="black", alpha=0.5)
    plt.xlabel("Collapse Factor")
    plt.ylabel("Average Absolute Error in % Change Predictions (percentage points)")
    plt.title("Error in % Change Predictions by Collapsed Dataset Models Varying Collapse Factor")
    plt.show()