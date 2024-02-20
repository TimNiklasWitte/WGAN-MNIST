from LoadDataframe import *
from matplotlib import pyplot as plt


def main():
    log_dir = "../logs"

    df = load_dataframe(log_dir)

    fig, axes = plt.subplots(nrows=30, ncols=10, figsize=(7, 20))

    for row_idx, epoch in enumerate(range(0, 300, 10)):
        imgs = df.loc[epoch, "generated imgs"]
        for col_idx in range(10):
            img = imgs[col_idx]
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].axis("off")
    
    plt.tight_layout()
    plt.savefig("../plots/GeneratedImgsWhileTraining.png", bbox_inches='tight')
    plt.show()
   

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")