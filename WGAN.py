import tensorflow as tf

from Generator import *
from Critic import *

class WGAN(tf.keras.Model):

    def __init__(self):
        super(WGAN, self).__init__()

        self.generator = Generator()
        self.critic = Critic()

        self.LAMBDA = 10

    @tf.function
    def train_step_critic(self, img_real):
        
        batch_size = img_real.shape[0]
        noise = tf.random.uniform(minval=-1, maxval=1, shape=(batch_size, self.generator.noise_dim))

        with tf.GradientTape() as tape:
            tape.watch(noise)

            #
            # Gradient penality
            #

            with tf.GradientTape(persistent=True) as tape_gradient_penality:
                tape_gradient_penality.watch(noise)
                img_fake = self.generator(noise, training=True)
                score_img_fake = self.critic(img_fake, training=True)

            gradient = tape_gradient_penality.gradient(score_img_fake, img_fake)
            gradient = tf.reshape(gradient, shape=(batch_size, 32,32))
    
            gradient_norm = tf.norm(gradient, axis=[-2,-1])

            gradient_penality = (gradient_norm - 1)**2
            gradient_penality = tf.math.reduce_mean(gradient_penality)
            

            #
            # Evaluation by critic
            #
            
            
            score_img_real = self.critic(img_real, training=True)
            
            #
            # Total loss
            #

            # Earthmover distance aka Wasserstein distance
            # gamma ("earth moving schedule" distribution hard to compute)
            # -> Kantorovich-Rubinstein duality (same equation for this distance)
            # sup tf.math.reduce_mean(rating_real) - tf.math.reduce_mean(rating_fake)
            # sup over lipschitz scalar function -> critic must be a lipschitz scalar function (limit on slope)
            #     -> enforce critic to be Lipschitz-continuous by gradient penality
            # approx sup by gradient ascent!

            # Want to maximize (gradient ascent) -> minus
            critic_default_loss = -(tf.math.reduce_mean(score_img_real) - tf.math.reduce_mean(score_img_fake))
            critic_loss = critic_default_loss + 10* gradient_penality


        # Update critic
        gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic.optimizer.apply_gradients(zip(gradients, self.critic.trainable_variables))

        self.critic.metric_loss.update_state(critic_loss)
        self.critic.metric_default_loss.update_state(critic_default_loss)
        self.critic.metric_gradient_penality.update_state(gradient_penality)

    @tf.function
    def train_step_generator(self, batch_size):

        noise = tf.random.uniform(minval=-1, maxval=1, shape=(batch_size, self.generator.noise_dim))

        with tf.GradientTape() as tape:
            img_fake = self.generator(noise, training=True)

            rating_fake = self.critic(img_fake, training=True)
            
            # Critic output meaning:
            # high <=> real
            # low <=> fake

            generator_loss = -tf.math.reduce_mean(rating_fake)
            # minus -> maximize -> it shall look more real :)
        
        # Update generator
        gradients = tape.gradient(generator_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gradients, self.generator.trainable_variables))

        self.generator.metric_loss.update_state(generator_loss)


