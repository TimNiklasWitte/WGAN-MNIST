import tensorflow as tf

from Generator import *
from Critic import *

class WGAN(tf.keras.Model):

    def __init__(self):
        super(WGAN, self).__init__()

        self.generator = Generator()
        self.critic = Critic()


    @tf.function
    def train_step(self, img_real):
        
        batch_size = img_real.shape[0]
        noise = tf.random.uniform(minval=-1, maxval=1, shape=(batch_size, self.generator.noise_dim))

        with tf.GradientTape(persistent=True) as tape:
            
            tape.watch(noise)
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch(noise)
                img_fake = self.generator(noise, training=True)

            gradient = tape1.gradient(img_fake, noise)
            gradient_norm = tf.norm(gradient, axis=-1)

            gradient_penality = (gradient_norm - 1)**2
            gradient_penality = tf.math.reduce_mean(gradient_penality)

            rating_fake = self.critic(img_fake, training=True)
            rating_real = self.critic(img_real, training=True)

            #
            # Gradient descent -> minimize these values
            #
            generator_loss = tf.math.reduce_mean(rating_fake)

            critic_default_loss = tf.math.reduce_mean(rating_real - rating_fake)
            critic_loss =  critic_default_loss + 10* gradient_penality


        # Update generator
        gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        # Update critic
        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))


        # Weight clipping
        # weight = 1
        # for w in self.critic.trainable_variables:
        #     w.assign(tf.clip_by_value(w, -weight, weight))

        #
        # Update metrices
        #
            
        self.generator.metric_loss.update_state(generator_loss)

        self.critic.metric_loss.update_state(critic_loss)
        self.critic.metric_default_loss.update_state(critic_default_loss)
        self.critic.metric_gradient_penality.update_state(gradient_penality)
