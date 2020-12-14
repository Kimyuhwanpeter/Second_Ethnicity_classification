# -*- coding:utf-8 -*-
import tensorflow as tf

def Conv1_max(input, filters, kernel_size, strides, padding, weight_decay, use_bias=False):

    h = tf.keras.layers.ZeroPadding2D((3,3))(input)
    h = tf.keras.layers.Conv2D(filters=filters,
                               kernel_size=kernel_size,
                               strides=strides,
                               padding=padding,
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 112 x 112 x 64

    h = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="same")(h)

    return h

def block_1(input, i, weight_decay):

    if i == 0:
        h = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=1,
                                    strides=1,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=3,
                                    strides=1,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=256,
                                    kernel_size=1,
                                    strides=1,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)
    else:
        h = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=1,
                                    strides=1,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=64,
                                    kernel_size=3,
                                    strides=1,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=256,
                                    kernel_size=1,
                                    strides=1,
                                    padding="same",
                                    use_bias=False,
                                    kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        u = h

        h = tf.keras.layers.GlobalAveragePooling2D()(h)
        h = tf.keras.layers.Dense(256 // 16)(h)
        h = tf.keras.layers.ReLU()(h)
        h = tf.keras.layers.Dense(256)(h)
        h = tf.keras.layers.Activation("sigmoid")(h)
        h = tf.expand_dims(h, 1)
        h = tf.expand_dims(h, 1)

        h = u * h
        h = h + input

    return h

def block_2(input, i, weight_decay):

    if i == 0:
        h = tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=1,
                                   strides=2,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=3,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=512,
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)
    else:
        h = tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=128,
                                   kernel_size=3,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=512,
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        u = h

        h = tf.keras.layers.GlobalAveragePooling2D()(h)
        h = tf.keras.layers.Dense(512 // 16)(h)
        h = tf.keras.layers.ReLU()(h)
        h = tf.keras.layers.Dense(512)(h)
        h = tf.keras.layers.Activation("sigmoid")(h)
        h = tf.expand_dims(h, 1)
        h = tf.expand_dims(h, 1)

        h = u * h
        h = h + input

    return h

def block_3(input, i, weight_decay):

    if i == 0:
        h = tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=1,
                                   strides=2,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=3,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=1024,
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)
    else:
        h = tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=256,
                                   kernel_size=3,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=1024,
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        u = h

        h = tf.keras.layers.GlobalAveragePooling2D()(h)
        h = tf.keras.layers.Dense(1024 // 16)(h)
        h = tf.keras.layers.ReLU()(h)
        h = tf.keras.layers.Dense(1024)(h)
        h = tf.keras.layers.Activation("sigmoid")(h)
        h = tf.expand_dims(h, 1)
        h = tf.expand_dims(h, 1)

        h = u * h
        h = h + input

    return h

def block_4(input, i, weight_decay):

    if i == 0:
        h = tf.keras.layers.Conv2D(filters=512,
                                   kernel_size=1,
                                   strides=2,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=512,
                                   kernel_size=3,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=2048,
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)
    else:
        h = tf.keras.layers.Conv2D(filters=512,
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=512,
                                   kernel_size=3,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=2048,
                                   kernel_size=1,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
        h = tf.keras.layers.BatchNormalization()(h)
        u = h

        h = tf.keras.layers.GlobalAveragePooling2D()(h)
        h = tf.keras.layers.Dense(2048 // 16)(h)
        h = tf.keras.layers.ReLU()(h)
        h = tf.keras.layers.Dense(2048)(h)
        h = tf.keras.layers.Activation("sigmoid")(h)
        h = tf.expand_dims(h, 1)
        h = tf.expand_dims(h, 1)

        h = u * h
        h = h + input

    return h

def se_resnet_50(input_shape=(224, 224, 3), weight_decay=0.000005):

    h = inputs = tf.keras.Input(input_shape)
    # 3,4,6,3
    #First_stage
    h = Conv1_max(input=h,
                  filters=64,
                  kernel_size=7,
                  strides=2,
                  padding="valid",
                  weight_decay=weight_decay)   # 56 x 56 x 64
    for i in range(3):
        h = block_1(h, i, weight_decay)

    for i in range(4):
        h = block_2(h, i, weight_decay)

    for i in range(6):
        h = block_3(h, i, weight_decay)

    for i in range(3):
        h = block_4(h, i, weight_decay)

    h = tf.keras.layers.GlobalAveragePooling2D()(h)
    h = tf.keras.layers.Dense(1)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)