import pickle
from matplotlib import pyplot as plt
from bert import get_p_diffs
from matplotlib.lines import Line2D

dir_loss = "experiment_data/embed/embed_losses/"

def demonstrate_embed_experiment(original_embed_size, new_embed_size):
    with open(f"{dir_loss}2_{original_embed_size}_odom.pkl", "rb") as file:
        old_d_old_m = pickle.load(file)
    with open(f"{dir_loss}2_{original_embed_size}_ndom.pkl", "rb") as file:
        new_d_old_m = pickle.load(file)
    with open(f"{dir_loss}2_{new_embed_size}_odnm.pkl", "rb") as file:
        old_d_new_m = pickle.load(file)
    with open(f"{dir_loss}2_{new_embed_size}_ndnm.pkl", "rb") as file:
        new_d_new_m = pickle.load(file)

    # Converages before 100
    plt.plot(new_d_old_m[:100], label="New Dataset", color="C1")
    plt.plot(old_d_old_m[:100], label="Original Dataset", color="C0")
    plt.plot(new_d_new_m[:100], color="C1", linestyle="--")
    plt.plot(old_d_new_m[:100], color="C0", linestyle="--")
    # https://stackoverflow.com/questions/51054529/manipulate-linestyle-in-matplotlib-legend
    line = Line2D([0,1],[0,1],linestyle='-', color='C1')
    line2 = Line2D([0,1],[0,1],linestyle='-', color='C0')
    line3 = Line2D([0,1],[0,1],linestyle='-', color='black')
    line4 = Line2D([0,1],[0,1],linestyle='--', color='black')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend([line, line2, line3, line4], ["Collapsed Dataset", "Original Dataset", "Old Model", "New Model"])
    plt.show()

    p_diffs_new = get_p_diffs(new_d_old_m[:100], new_d_new_m[:100])
    p_diffs_old = get_p_diffs(old_d_old_m[:100], old_d_new_m[:100])
    plt.plot(p_diffs_new, label="Collapsed Dataset", color="C1")
    plt.plot(p_diffs_old, label="Original Dataset", color="C0")
    plt.xlabel("Epoch")
    plt.ylabel("Difference in Loss Between Models (%)")
    plt.legend()
    plt.show()

demonstrate_embed_experiment(128, 256)