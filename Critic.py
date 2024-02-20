import tensorflow as tf

class Critic(tf.keras.Model):

    def __init__(self):
        super(Critic, self).__init__()

        self.layer_list = [
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2,2), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LeakyReLU(),
            
            tf.keras.layers.Flatten(),

            # Output meaning:
            # high <=> real
            # low <=> fake
            tf.keras.layers.Dense(1, activation=None)
        ]

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
        self.metric_default_loss = tf.keras.metrics.Mean(name="default_loss")
        self.metric_gradient_penality = tf.keras.metrics.Mean(name="gradient_penality")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            if isinstance(layer, tf.keras.layers.BatchNormalization) or isinstance(layer, tf.keras.layers.Dropout):
                x = layer(x, training)
            else:
                x = layer(x)
        return x
