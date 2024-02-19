from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    
    sns.lineplot(data=df.loc[:, ["generator loss"]], ax=axes[0], legend=None, palette=['red'])
    axes[0].set_title("Generator loss")


    sns.lineplot(data=df.loc[:, ["critic default loss"]], ax=axes[1], legend=None, palette=['blue'])
    axes[1].set_title("Critic loss")
    
    sns.lineplot(data=df.loc[:, ["critic gradient penality"]], ax=axes[2], legend=None, palette=['tab:blue'])
    axes[2].set_title("Critic gradient penality")

    # grid
    for ax in axes.flatten():
        ax.grid()

    plt.tight_layout()
    plt.savefig("../plots/Loss_GradientPenality.png")
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")