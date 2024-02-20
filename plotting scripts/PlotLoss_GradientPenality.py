from LoadDataframe import *
from matplotlib import pyplot as plt

import seaborn as sns

def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)

    fig, axes = plt.subplots(1, 3, figsize=(15,5))

    df2 = df.rename(columns={'generator loss':'generator', "critic default loss": "critic"})
    plot = sns.lineplot(data=df2.loc[:, ["generator", "critic"]], ax=axes[0], dashes=False, palette=["red", "blue"])
    axes[0].set_title("Generator and critic loss")
    axes[0].set_ylabel("Loss")
    axes[0].legend(loc="upper right")
    plot.set(ylim=(None, 2))

    df2 = df.rename(columns={'critic score img real':'real img', "critic score img fake": "fake img"})
    sns.lineplot(data=df2.loc[:, ["real img", "fake img"]], ax=axes[1], dashes=False)
    axes[1].set_title("Critic score")
    axes[1].set_ylabel("Critic score")
    
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