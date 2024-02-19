import sys
sys.path.append("../")

import tensorflow as tf

from LoadDataframe import *
from matplotlib import pyplot as plt


from WGAN import *


def main():

    gan = WGAN()

    # Build 
    gan.critic.build(input_shape=(1, 32, 32, 1))
    gan.generator.build(input_shape=(1, gan.generator.noise_dim))
    
    # Get overview of number of parameters
    gan.critic.summary()
    gan.generator.summary()

    gan.load_weights(f"../saved_models/trained_weights_150").expect_partial()

    noise = tf.random.normal(shape=(100, gan.generator.noise_dim))

    imgs = gan.generator(noise)

    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))

    imgs = tf.reshape(imgs, shape=(10, 10, 32, 32))
    for row_idx in range(10):
        for col_idx in range(10):
            img = imgs[row_idx, col_idx, :, :]
            axes[row_idx, col_idx].imshow(img)
            axes[row_idx, col_idx].axis("off")

    
    plt.tight_layout()
    plt.savefig("../plots/GenerateImgs.png", bbox_inches='tight')
    plt.show()
   

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")