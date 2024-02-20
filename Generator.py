import tensorflow as tf

class Generator(tf.keras.Model):

    def __init__(self):
        super(Generator, self).__init__()

        self.layer_list = [
            tf.keras.layers.Dense(4*4*64, use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Reshape((4, 4, 64)),

            tf.keras.layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LeakyReLU(),

            tf.keras.layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='same', use_bias=False, activation='tanh')
        ]

        self.metric_loss = tf.keras.metrics.Mean(name="loss")
        
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

        self.noise_dim = 100

    @tf.function
    def call(self, x, training=False):
        for layer in self.layer_list:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                x = layer(x, training)
            else:
                x = layer(x)
        return x
    

