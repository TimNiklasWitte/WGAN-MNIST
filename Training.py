import tensorflow as tf
import tensorflow_datasets as tfds
import tqdm
import datetime

from WGAN import *

NUM_EPOCHS = 150
BATCH_SIZE = 32

def main():

    #
    # Load dataset
    #   

    train_ds = tfds.load("mnist", split="train+test", as_supervised=True)

    train_ds = train_ds.apply(prepare_data)


    #
    # Logging
    #

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_path = f"logs/{current_time}"
    train_summary_writer = tf.summary.create_file_writer(file_path)

    #
    # Initialize model.
    #

    gan = WGAN()

    # Build 
    gan.critic.build(input_shape=(1, 32, 32, 1))
    gan.generator.build(input_shape=(1, gan.generator.noise_dim))
    
    # Get overview of number of parameters
    gan.critic.summary()
    gan.generator.summary()
    
    #
    # Init noise
    # Take always the same noise for generating images
    # -> evaluation!
    #
    num_generated_imgs = 32
    noise = tf.random.uniform(minval=-1, maxval=1, shape=(num_generated_imgs, gan.generator.noise_dim))

    #
    # Train loop
    #
    for epoch in range(1, NUM_EPOCHS + 1):
            
        print(f"Epoch {epoch}")

        for i, img_real in enumerate(tqdm.tqdm(train_ds, position=0, leave=True)): 

            if i % 5 == 0:
                gan.train_step_generator(BATCH_SIZE)
                gan.train_step_critic(img_real)
            else:
                gan.train_step_critic(img_real)

        log(train_summary_writer, gan, noise, epoch)

        if epoch % 50 == 0:
            # Save model (its parameters)
            gan.save_weights(f"./saved_models/trained_weights_{epoch}", save_format="tf")


def log(train_summary_writer, gan, noise, epoch):

    generator_loss = gan.generator.metric_loss.result()
    
    #
    # Generate images
    #

    generated_imgs = gan.generator(noise, training=False)
  
    #
    # Write to TensorBoard
    #
    num_generated_imgs = noise.shape[0]
    with train_summary_writer.as_default():
        tf.summary.scalar(f"generator_loss", generator_loss, step=epoch)

        for metric in gan.critic.metrics:
            tf.summary.scalar(f"critic_{metric.name}", metric.result(), step=epoch)

        tf.summary.image(name="generated_imgs",data = generated_imgs, step=epoch, max_outputs=num_generated_imgs)
        
    #
    # Output
    #
    print(f"generator_loss: {generator_loss}")
    for metric in gan.critic.metrics:
        print(f"critic_{metric.name}: {metric.result()}")

    
    #
    # Reset metrices
    #

    gan.generator.metric_loss.reset_states()

    for metric in gan.critic.metrics:
        metric.reset_states()

 
def prepare_data(dataset):

    #dataset = dataset.filter(lambda img, label: label == 0) # only '0' digits

    # Remove label
    dataset = dataset.map(lambda img, label: img)

    dataset = dataset.map(lambda img: tf.image.resize(img, [32,32]) )

    # Convert data from uint8 to float32
    dataset = dataset.map(lambda img: tf.cast(img, tf.float32) )

    #Sloppy input normalization, just bringing image values from range [0, 255] to [-1, 1]
    dataset = dataset.map(lambda img: (img/128.)-1. )

    # Cache
    dataset = dataset.cache()
    
    #
    # Shuffle, batch, prefetch
    #
    dataset = dataset.shuffle(1000)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

    return dataset

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("KeyboardInterrupt received")