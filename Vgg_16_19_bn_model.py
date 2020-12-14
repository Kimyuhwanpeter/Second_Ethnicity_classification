# -*- coding:utf-8 -*-
import tensorflow as tf

Conv2D = tf.keras.layers.Conv2D
Maxpool2D = tf.keras.layers.MaxPool2D
BatchNorm = tf.keras.layers.BatchNormalization
ReLU = tf.keras.layers.ReLU

def conv_bn_relu(input, filters, kernel_size, strides, padding, use_bias, weight_decay):

    if use_bias == True:
        h = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   use_bias=use_bias,
                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
        h = ReLU()(h)
    else:
        h = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   strides=strides,
                   padding=padding,
                   use_bias=use_bias,
                   kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
        h = BatchNorm()(h)
        h = ReLU()(h)

    return h

def Vgg16_bn(input_shape=(224, 224, 3), weight_decay=0.000005):

    h = inputs = tf.keras.Input(input_shape)

    h = conv_bn_relu(h,64,3,1,"same",False,weight_decay)    # 224 x 224 x 64
    h = conv_bn_relu(h,64,3,1,"same",False,weight_decay)    # 224 x 224 x 64

    h = Maxpool2D(pool_size=(3,3), strides=2, padding="same")(h)    # 112 x 112 x 64

    h = conv_bn_relu(h,128,3,1,"same",False,weight_decay)    # 112 x 112 x 128
    h = conv_bn_relu(h,128,3,1,"same",False,weight_decay)    # 112 x 112 x 128

    h = Maxpool2D(pool_size=(3,3), strides=2, padding="same")(h)    # 56 x 56 x 128

    h = conv_bn_relu(h,256,3,1,"same",False,weight_decay)    # 56 x 56 x 256
    h = conv_bn_relu(h,256,3,1,"same",False,weight_decay)    # 56 x 56 x 256
    h = conv_bn_relu(h,256,3,1,"same",False,weight_decay)    # 56 x 56 x 256

    h = Maxpool2D(pool_size=(3,3), strides=2, padding="same")(h)    # 28 x 28 x 256

    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 28 x 28 x 512
    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 28 x 28 x 512
    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 28 x 28 x 512

    h = Maxpool2D(pool_size=(3,3), strides=2, padding="same")(h)    # 14 x 14 x 256

    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 14 x 14 x 512
    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 14 x 14 x 512
    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 14 x 14 x 512

    h = Maxpool2D(pool_size=(3,3), strides=2, padding="same")(h)    # 7 x 7 x 512

    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(4096)(h)
    h = tf.keras.layers.Dense(4096)(h)
    h = tf.keras.layers.Dense(1)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

def Vgg19_bn(input_shape=(224, 224, 3), weight_decay=0.000005):

    h = inputs = tf.keras.Input(input_shape)

    h = conv_bn_relu(h,64,3,1,"same",False,weight_decay)    # 224 x 224 x 64
    h = conv_bn_relu(h,64,3,1,"same",False,weight_decay)    # 224 x 224 x 64

    h = Maxpool2D(pool_size=(3,3), strides=2, padding="same")(h)    # 112 x 112 x 64

    h = conv_bn_relu(h,128,3,1,"same",False,weight_decay)    # 112 x 112 x 128
    h = conv_bn_relu(h,128,3,1,"same",False,weight_decay)    # 112 x 112 x 128

    h = Maxpool2D(pool_size=(3,3), strides=2, padding="same")(h)    # 56 x 56 x 128

    h = conv_bn_relu(h,256,3,1,"same",False,weight_decay)    # 56 x 56 x 256
    h = conv_bn_relu(h,256,3,1,"same",False,weight_decay)    # 56 x 56 x 256
    h = conv_bn_relu(h,256,3,1,"same",False,weight_decay)    # 56 x 56 x 256
    h = conv_bn_relu(h,256,3,1,"same",False,weight_decay)    # 56 x 56 x 256

    h = Maxpool2D(pool_size=(3,3), strides=2, padding="same")(h)    # 28 x 28 x 256

    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 28 x 28 x 512
    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 28 x 28 x 512
    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 28 x 28 x 512
    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 28 x 28 x 512

    h = Maxpool2D(pool_size=(3,3), strides=2, padding="same")(h)    # 14 x 14 x 512

    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 14 x 14 x 512
    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 14 x 14 x 512
    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 14 x 14 x 512
    h = conv_bn_relu(h,512,3,1,"same",False,weight_decay)    # 14 x 14 x 512

    h = Maxpool2D(pool_size=(3,3), strides=2, padding="same")(h)    # 7 x 7 x 512

    h = tf.keras.layers.Flatten()(h)
    h = tf.keras.layers.Dense(4096)(h)
    h = tf.keras.layers.Dense(4096)(h)
    h = tf.keras.layers.Dense(1)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)