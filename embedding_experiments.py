import time
from transformers import BertConfig
from utils import get_vocab_size, num_params
import pickle
from matplotlib import pyplot as plt
from transformers import BertForMaskedLM
import os
import numpy as np
from bert import MAX_LEN, pad_and_mask, add_mlm_masking, train, get_p_diffs, get_p_diff

"""
------------------------------------------------------------------------------------------------------------------------
This script is used for conducting the experiment on the technique's effectiveness using various embedding size changes.
------------------------------------------------------------------------------------------------------------------------
"""

def run_embedding_test(new_embed_size, original_embed_size, new_vocab_size, 
                       original_vocab_size, dir_loss, dir_model, collapse_factor, 
                       original_dataset_ids, original_dataset_mask, original_dataset_labels,
                       original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels,
                       new_dataset_ids, new_dataset_mask, new_dataset_labels,
                       new_dataset_val_ids, new_dataset_val_mask, new_dataset_val_labels,
                       epochs=50):
    """
    Run an experiment measuring the technique's effectiveness on a particular embedding size change.
    
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
        
        epochs: Number of epochs to run
    """
    
    # Prepare the configs for each of the 4 models

    # Original vocab, original architecture
    bert_config_1 = BertConfig(
        vocab_size=original_vocab_size,
        hidden_size=original_embed_size,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=MAX_LEN,
    )

    # New vocab, original architecture
    bert_config_2 = BertConfig(
        vocab_size=new_vocab_size,
        hidden_size=original_embed_size,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=MAX_LEN,
    )

    # Original vocab, new architecture
    bert_config_3 = BertConfig(
        vocab_size=original_vocab_size,
        hidden_size=new_embed_size,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=MAX_LEN,
    )

    # New vocab, new architecture
    bert_config_4 = BertConfig(
        vocab_size=new_vocab_size,
        hidden_size=new_embed_size,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=512,
        max_position_embeddings=MAX_LEN,
    )

    # Calulate the total parameters of each model with the collapsed dataset vocabulary size
    total_params = num_params(BertForMaskedLM(bert_config_2))
    print(f"Original M parameters: {total_params}")
    total_params = num_params(BertForMaskedLM(bert_config_4))
    print(f"New M parameters: {total_params}")

    # For the original MODEL, only need to train once
    if not os.path.exists(f"{dir_loss}{collapse_factor}_{original_embed_size}_odom.pkl"):
        # Create and train the old dataset, old model combination
        model = BertForMaskedLM(bert_config_1)
        _, old_d_old_m = train(original_dataset_ids, original_dataset_mask, original_dataset_labels,
                               original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels, model, epochs)
        # Save the model and loss
        with open(f"{dir_loss}{collapse_factor}_{original_embed_size}_odom.pkl", "wb") as file:
            pickle.dump(old_d_old_m, file)
        model.save_pretrained(f"{dir_model}{collapse_factor}_{original_embed_size}_odom")
    else:
        print("Using saved model for old dataset")
        with open(f"{dir_loss}{collapse_factor}_{original_embed_size}_odom.pkl", "rb") as file:
            old_d_old_m = pickle.load(file)

    # For the original MODEL, only need to train once
    if not os.path.exists(f"{dir_loss}{collapse_factor}_{original_embed_size}_ndom.pkl"):
        # Create and train the new dataset, old model combination
        model = BertForMaskedLM(bert_config_2)
        _, new_d_old_m = train(new_dataset_ids, new_dataset_mask, new_dataset_labels,
                               new_dataset_val_ids, new_dataset_val_mask, new_dataset_val_labels, model, epochs)
        # Save the model and loss
        with open(f"{dir_loss}{collapse_factor}_{original_embed_size}_ndom.pkl", "wb") as file:
            pickle.dump(new_d_old_m, file)
        model.save_pretrained(f"{dir_model}{collapse_factor}_{original_embed_size}_ndom")
    else:
        print("Using saved model for new dataset")
        with open(f"{dir_loss}{collapse_factor}_{original_embed_size}_ndom.pkl", "rb") as file:
            new_d_old_m = pickle.load(file)

    # Create and train the old dataset, new model combination
    model = BertForMaskedLM(bert_config_3)
    _, old_d_new_m = train(original_dataset_ids, original_dataset_mask, original_dataset_labels,
                           original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels, model, epochs)
    # Save the model and loss
    with open(f"{dir_loss}{collapse_factor}_{new_embed_size}_odnm.pkl", "wb") as file:
        pickle.dump(old_d_new_m, file)
    model.save_pretrained(f"{dir_model}{collapse_factor}_{new_embed_size}_odnm")

    # Create and train the new dataset, new model combination
    model = BertForMaskedLM(bert_config_4)
    _, new_d_new_m = train(new_dataset_ids, new_dataset_mask, new_dataset_labels,
                           new_dataset_val_ids, new_dataset_val_mask, new_dataset_val_labels, model, epochs)
    # Save the model and loss
    with open(f"{dir_loss}{collapse_factor}_{new_embed_size}_ndnm.pkl", "wb") as file:
        pickle.dump(new_d_new_m, file)
    model.save_pretrained(f"{dir_model}{collapse_factor}_{new_embed_size}_ndnm")
    
    # Preliminary plotting to assess how each experiment looks
    # Plotting all losses:
    plt.plot(new_d_old_m, label="New Dataset Original Model")
    plt.plot(old_d_old_m, label="Original Dataset Original Model")
    plt.plot(new_d_new_m, label="New Dataset New Model")
    plt.plot(old_d_new_m, label="Original Dataset New Model")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()
    
    # Plotting percent differencces between each model
    # Calculate the the percent difference between models
    p_diffs_new = get_p_diffs(new_d_old_m[:len(new_d_new_m)], new_d_new_m)# The old model ones may be trained on more epochs, need to compare at the same epoch
    p_diffs_old = get_p_diffs(old_d_old_m[:len(old_d_new_m)], old_d_new_m)
    plt.plot(p_diffs_new, label="Both on New Dataset")
    plt.plot(p_diffs_old, label="Both on Original Dataset")
    plt.xlabel("Epoch")
    plt.ylabel("Percent Diff")
    plt.legend()
    plt.show()


def bar_graph(collapse_factor, original_embed_size, new_embed_sizes, dir_loss):
    """
    Create a bar plot demonstrating the technique's effectiveness using various embedding size changes.
    
    Args:
        collapse_factor: The collapse factor used in the form 2 for 50% reduction. Used to find the loss files.
        original_embed_size: The "old" embeding dimensionality. Used to find the loss file.
        new_embed_sizes: List of the "new" embeding dimensionalities. Used to find the loss files.
        dir_loss: Directory where the embedding experiment losses are stored
    """
    # Load the original model losses
    with open(f"{dir_loss}{collapse_factor}_{original_embed_size}_odom.pkl", "rb") as file:
        old_d_old_m = pickle.load(file)
    with open(f"{dir_loss}{collapse_factor}_{original_embed_size}_ndom.pkl", "rb") as file:
        new_d_old_m = pickle.load(file)

    # https://www.geeksforgeeks.org/bar-plot-in-matplotlib/
    # https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html#sphx-glr-gallery-lines-bars-and-markers-barchart-py
    _, ax = plt.subplots()
    width = 0.25
    x = np.arange(len(new_embed_sizes)) * 0.55 # reduces gap
    
    old_data_diffs = []
    new_data_diffs = []

    # Loop through each trial embed size
    for new_embed_size in new_embed_sizes:
        # Load the new model losses
        with open(f"{dir_loss}{collapse_factor}_{new_embed_size}_odnm.pkl", "rb") as file:
            old_d_new_m = pickle.load(file)
        with open(f"{dir_loss}{collapse_factor}_{new_embed_size}_ndnm.pkl", "rb") as file:
            new_d_new_m = pickle.load(file)
        # Calculate the the percent difference between models
        old_data_diffs.append(get_p_diff(min(old_d_old_m[:len(old_d_new_m)]), min(old_d_new_m)))
        new_data_diffs.append(get_p_diff(min(new_d_old_m[:len(new_d_new_m)]), min(new_d_new_m)))
    
    # Create the bars
    bar1 = ax.bar(x - width/2, old_data_diffs, width, label='Original Dataset', color='C0', edgecolor='black')
    bar2 = ax.bar(x + width/2, new_data_diffs, width, label='Collapsed Dataset', color='C1', edgecolor='black')
    
    # Add labels
    ax.set_xlabel('Embedding Size')
    ax.set_ylabel('Difference in Loss Between Models (%)')
    ax.set_xticks(x, [str(size) for size in new_embed_sizes])
    ax.legend()
    
    # Add text to the bars
    # https://stackoverflow.com/questions/40489821/how-to-write-text-above-the-bars-on-a-bar-plot-python
    for rect in bar1 + bar2:
        height = rect.get_height()
        plt.text(rect.get_x() + rect.get_width() / 2, height, f'{height:.2f}', ha='center', va='bottom')
    
    plt.show()



if __name__ == "__main__":
    # Specify collapse factor and experiment directories
    collapse_factor = 2
    dir_loss = "experiment_data/embed/embed_losses/"
    dir_model = "experiment_data/embed/embed_models/"

    # Get both vocab sizes
    try:
        original_vocab_size = get_vocab_size("base_tokenizer")
    except Exception as e:
        raise Exception(f"Could not find the original tokenizer. Make sure it is downlaoded from HuggingFace and named 'base_tokenizer': {e}")
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
    
    for new_embed_size in [32, 64, 256, 512]:
        if new_embed_size == 32: # takes longer to converge for some reason
            epochs = 200
        else:
            epochs = 150
        run_embedding_test(128, new_embed_size, new_vocab_size, 
                           original_vocab_size, dir_loss, dir_model, collapse_factor, 
                           original_dataset_ids, original_dataset_mask, original_dataset_labels,
                           original_dataset_val_ids, original_dataset_val_mask, original_dataset_val_labels,
                           new_dataset_ids, new_dataset_mask, new_dataset_labels,
                           new_dataset_val_ids, new_dataset_val_mask, new_dataset_val_labels,
                           epochs=epochs)

    bar_graph(collapse_factor, 128, [32, 64, 256, 512], dir_loss)
