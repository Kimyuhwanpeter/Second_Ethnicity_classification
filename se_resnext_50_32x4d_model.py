import tensorflow as tf

class Group_Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides, padding, use_bias, weight_decay, n_group):
        super(Group_Conv2D, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = self._strides = strides
        self.padding = padding
        self.use_bias = use_bias
        self.weight_decay = weight_decay
        self.n_group = n_group

    def build(self, input_shape):
        input_shape = input_shape
        input_channel = input_shape[3]
        self._strides = [1, self._strides, self._strides, 1]
        kernel_shape = (self.kernel_size, self.kernel_size) + (input_channel // self.n_group, self.filters)
        
        self.kernel = self.add_weight(
            name="kernel",
            shape=kernel_shape,
            initializer=tf.keras.initializers.glorot_uniform(),
            regularizer=tf.keras.regularizers.l2(self.weight_decay),
            trainable=True,
            dtype=tf.float32)
        if self.use_bias:
            self.bias = self.add_weight(
            name="bias",
            shape=(self.filters,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
            dtype=tf.float32)
        else:
            self.bias = None

        self.groupConv = lambda i, k: tf.nn.conv2d(i,
                                                   k,
                                                   strides=self._strides,
                                                   padding=self.padding,
                                                   dilations=(1,1))

    def call(self, inputs):

        if self.n_group == 1:
            outputs = self.groupConv(inputs, self.kernel)
        else:
            inputGroups = tf.split(axis=3, num_or_size_splits=self.n_group, value=inputs)
            weightGroups = tf.split(axis=3, num_or_size_splits=self.n_group, value=self.kernel)
            convGroups = [self.groupConv(i, k) for i, k in zip(inputGroups, weightGroups)]
            outputs = tf.concat(convGroups, 3)
        if self.use_bias:
            outputs = tf.nn.bias_add(outputs, self.bias)

        return outputs
        

def block(input, filters, i, weight_decay, reduction, se):

    if i == 0:
        if reduction == 0:
            h = tf.keras.layers.Conv2D(filters=filters,
                                        kernel_size=1,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
            h = tf.keras.layers.BatchNormalization()(h)
            h = tf.keras.layers.ReLU()(h)
        else:
            h = tf.keras.layers.Conv2D(filters=filters,
                                        kernel_size=1,
                                        strides=1,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
            h = tf.keras.layers.BatchNormalization()(h)
            h = tf.keras.layers.ReLU()(h)

        h = Group_Conv2D(filters=filters,
                            kernel_size=3,
                            strides=1,
                            padding="SAME",
                            use_bias=False,
                            weight_decay=weight_decay,
                            n_group=32)(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=filters*2,
                                    kernel_size=1,
                                    strides=1,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)
    else:
        h = tf.keras.layers.Conv2D(filters=filters,
                                    kernel_size=1,
                                    strides=1,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = Group_Conv2D(filters=filters,
                         kernel_size=3,
                         strides=1,
                         padding="SAME",
                         use_bias=False,
                         weight_decay=weight_decay,
                         n_group=32)(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=filters*2,
                                    kernel_size=1,
                                    strides=1,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        if se:
            u = h

            h = tf.keras.layers.GlobalAveragePooling2D()(h)
            h = tf.keras.layers.Dense(filters*2 // 16)(h)
            h = tf.keras.layers.ReLU()(h)
            h = tf.keras.layers.Dense(filters*2)(h)
            h = tf.keras.layers.Activation("sigmoid")(h)
            h = tf.expand_dims(h, 1)
            h = tf.expand_dims(h, 1)

            h = u * h
            h = h + input

            h = tf.keras.layers.ReLU()(h)
        else:
            h = tf.keras.layers.ReLU()(h + input)

    return h

def se_resnext_50_32x4d(input_shape=(224, 224, 3), weight_decay=0.000005):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 112 x 112 x 64

    h = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="same")(h)    # 56 x 56 x 64

    for i in range(3):
        h = block(h, 128, i, weight_decay, i + 1, True)   # 56 x 56 x 256

    for i in range(4):
        h = block(h, 256, i, weight_decay, i, True)   # 28 x 28 x 512

    for i in range(6):
        h = block(h, 512, i, weight_decay, i, True)   # 14 x 14 x 1024

    for i in range(3):
        h = block(h, 1024, i, weight_decay, i, True)  # 7 x 7 x 2048

    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.keras.layers.Dense(1000)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def resnext_50_32x4d(input_shape=(224, 224, 3), weight_decay=0.000005):

    h = inputs = tf.keras.Input(input_shape)

    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=7,
                               strides=2,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 112 x 112 x 64

    h = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="same")(h)    # 56 x 56 x 64

    for i in range(3):
        h = block(h, 128, i, weight_decay, i + 1, False)   # 56 x 56 x 256

    for i in range(4):
        h = block(h, 256, i, weight_decay, i, False)   # 28 x 28 x 512

    for i in range(6):
        h = block(h, 512, i, weight_decay, i, False)   # 14 x 14 x 1024

    for i in range(3):
        h = block(h, 1024, i, weight_decay, i, False)  # 7 x 7 x 2048

    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.keras.layers.Dense(1000)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

model = se_resnext_50_32x4d()
model.summary()

model2 = resnext_50_32x4d()
model2.summary()