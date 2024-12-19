import pickle
from matplotlib import pyplot as plt
from bert import get_p_diffs
from matplotlib.lines import Line2D

"""
----------------------------------------------------------------------------------------------------
This script is used for creating plots to demonstrate the technique using one embedding size change.
----------------------------------------------------------------------------------------------------
"""

def demonstrate_embed_experiment(original_embed_size, new_embed_size, dir_loss):
    """
    Create plots to demonstrate the technique using one embedding size change.
    
    Args:
        new_embed_size: Embedding dimensionality of the "new" model. Used to grab the a loss file from the embedding experiments.
        original_embed_size: Embedding dimensionality of the "old" model. Used to grab the a loss file from the embedding experiments.
        dir_loss: The directory where the embedding experiment losses are stored
    """
    # Load the four models for each architecture and dataset combination
    try:
        with open(f"{dir_loss}2_{original_embed_size}_odom.pkl", "rb") as file:
            old_d_old_m = pickle.load(file)
        with open(f"{dir_loss}2_{original_embed_size}_ndom.pkl", "rb") as file:
            new_d_old_m = pickle.load(file)
        with open(f"{dir_loss}2_{new_embed_size}_odnm.pkl", "rb") as file:
            old_d_new_m = pickle.load(file)
        with open(f"{dir_loss}2_{new_embed_size}_ndnm.pkl", "rb") as file:
            new_d_new_m = pickle.load(file)
    except Exception as e:
        raise Exception(f"Could not find the embedding experiment losses. Make sure to run the embedding experiment first: {e}")

    # --- Plotting ---
    # Plot each model loss individually
    plt.plot(new_d_old_m[:100], label="New Dataset", color="C1") # Converages before 100
    plt.plot(old_d_old_m[:100], label="Original Dataset", color="C0")
    plt.plot(new_d_new_m[:100], color="C1", linestyle="--")
    plt.plot(old_d_new_m[:100], color="C0", linestyle="--")
    # Create lines for the legend
    # https://stackoverflow.com/questions/51054529/manipulate-linestyle-in-matplotlib-legend
    line = Line2D([0,1],[0,1],linestyle='-', color='C1')
    line2 = Line2D([0,1],[0,1],linestyle='-', color='C0')
    line3 = Line2D([0,1],[0,1],linestyle='-', color='black')
    line4 = Line2D([0,1],[0,1],linestyle='--', color='black')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend([line, line2, line3, line4], ["Collapsed Dataset", "Original Dataset", "Old Model", "New Model"])
    plt.show()

    # Plot the percent difference between models on each dataset
    # Get the percent differences
    p_diffs_new = get_p_diffs(new_d_old_m[:100], new_d_new_m[:100])
    p_diffs_old = get_p_diffs(old_d_old_m[:100], old_d_new_m[:100])
    # Plot the differences
    plt.plot(p_diffs_new, label="Collapsed Dataset", color="C1")
    plt.plot(p_diffs_old, label="Original Dataset", color="C0")
    # Add labels
    plt.xlabel("Epoch")
    plt.ylabel("Difference in Loss Between Models (%)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    dir_loss = "experiment_data/embed/embed_losses/"
    demonstrate_embed_experiment(128, 256, dir_loss)