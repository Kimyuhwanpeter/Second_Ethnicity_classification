import tensorflow as tf

def stem(input, weight_decay):

    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               strides=2,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 149 x 149 x 32

    h = tf.keras.layers.Conv2D(filters=32,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 147 x 147 x 32

    h = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=3,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # 147 x 147 x 64

    h_1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="valid")(h) # 73 x 73 x 64
    h_2 = tf.keras.layers.Conv2D(filters=96,
                               kernel_size=3,
                               strides=2,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)   # 73 x 73 x 96

    h = tf.concat([h_1, h_2], 3)    # 73 x 73 x 160

    h_1 = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=1,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)   # 73 x 73 x 64

    h_1 = tf.keras.layers.Conv2D(filters=96,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)   # 71 x 71 x 96

    h_2 = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=1,
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)   # 73 x 73 x 64

    h_2 = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(7,1),
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)   # 73 x 73 x 64

    h_2 = tf.keras.layers.Conv2D(filters=64,
                               kernel_size=(1,7),
                               strides=1,
                               padding="same",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)   # 73 x 73 x 64

    h_2 = tf.keras.layers.Conv2D(filters=96,
                               kernel_size=3,
                               strides=1,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)   # 71 x 71 x 96

    h = tf.concat([h_1, h_2], 3)    # 71 x 71 x 192

    h_1 = tf.keras.layers.Conv2D(filters=192,
                               kernel_size=3,
                               strides=2,
                               padding="valid",
                               use_bias=False,
                               kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)   # 35 x 35 x 192

    h_2 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="valid")(h) # 35 x 35 x 192

    h = tf.concat([h_1, h_2], 3)    # 35 x 35 x 384

    return h

def inception_A(input, weight_decay):

    h_1 = tf.keras.layers.AvgPool2D(pool_size=(3,3), strides=1, padding="same")(input)
    h_1 = tf.keras.layers.Conv2D(filters=96,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2 = tf.keras.layers.Conv2D(filters=96,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    h_3 = tf.keras.layers.Conv2D(filters=64,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    h_3 = tf.keras.layers.Conv2D(filters=96,
                                 kernel_size=3,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_3)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    h_4 = tf.keras.layers.Conv2D(filters=64,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)

    h_4 = tf.keras.layers.Conv2D(filters=96,
                                 kernel_size=3,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_4)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)

    h_4 = tf.keras.layers.Conv2D(filters=96,
                                 kernel_size=3,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_4)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)

    h = tf.concat([h_1, h_2, h_3, h_4], 3)  # 35 x 35 x 384

    return h

def inception_B(input, weight_decay):
    # Fall dataset 코랩으로 학습중인것 모델이나 loss를 다시 수정해야한다!! 기억해!!!!!
    h_1 = tf.keras.layers.AvgPool2D(pool_size=(3,3), strides=1, padding="same")(input)
    h_1 = tf.keras.layers.Conv2D(filters=128,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2 = tf.keras.layers.Conv2D(filters=384,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    h_3 = tf.keras.layers.Conv2D(filters=192,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    h_3 = tf.keras.layers.Conv2D(filters=224,
                                 kernel_size=(1,7),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_3)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    h_3 = tf.keras.layers.Conv2D(filters=256,
                                 kernel_size=(1,7),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_3)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    h_4 = tf.keras.layers.Conv2D(filters=192,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)

    h_4 = tf.keras.layers.Conv2D(filters=192,
                                 kernel_size=(1,7),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_4)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)

    h_4 = tf.keras.layers.Conv2D(filters=224,
                                 kernel_size=(7,1),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_4)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)

    h_4 = tf.keras.layers.Conv2D(filters=224,
                                 kernel_size=(1,7),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_4)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)

    h_4 = tf.keras.layers.Conv2D(filters=256,
                                 kernel_size=(7,1),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_4)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)

    h = tf.concat([h_1, h_2, h_3, h_4], 3)

    return h

def inception_C(input, weight_decay):

    h_1 = tf.keras.layers.AvgPool2D(pool_size=(3,3), strides=1, padding="same")(input)
    h_1 = tf.keras.layers.Conv2D(filters=256,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_1)
    h_1 = tf.keras.layers.BatchNormalization()(h_1)
    h_1 = tf.keras.layers.ReLU()(h_1)

    h_2 = tf.keras.layers.Conv2D(filters=256,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    h_3 = tf.keras.layers.Conv2D(filters=384,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    h_3_1 = tf.keras.layers.Conv2D(filters=256,
                                 kernel_size=(1,3),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_3)
    h_3_1 = tf.keras.layers.BatchNormalization()(h_3_1)
    h_3_1 = tf.keras.layers.ReLU()(h_3_1)

    h_3_2 = tf.keras.layers.Conv2D(filters=256,
                                 kernel_size=(3,1),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_3)
    h_3_2 = tf.keras.layers.BatchNormalization()(h_3_2)
    h_3_2 = tf.keras.layers.ReLU()(h_3_2)

    h_4 = tf.keras.layers.Conv2D(filters=384,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)

    h_4 = tf.keras.layers.Conv2D(filters=448,
                                 kernel_size=(1,3),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_4)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)

    h_4 = tf.keras.layers.Conv2D(filters=512,
                                 kernel_size=(3,1),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_4)
    h_4 = tf.keras.layers.BatchNormalization()(h_4)
    h_4 = tf.keras.layers.ReLU()(h_4)

    h_4_1 = tf.keras.layers.Conv2D(filters=256,
                                 kernel_size=(3,1),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_4)
    h_4_1 = tf.keras.layers.BatchNormalization()(h_4_1)
    h_4_1 = tf.keras.layers.ReLU()(h_4_1)

    h_4_2 = tf.keras.layers.Conv2D(filters=256,
                                 kernel_size=(1,3),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_4)
    h_4_2 = tf.keras.layers.BatchNormalization()(h_4_2)
    h_4_2 = tf.keras.layers.ReLU()(h_4_2)

    h = tf.concat([h_1, h_2, h_3_1, h_3_2, h_4_1, h_4_2], 3)

    return h

def reduction_A(input, weight_decay):

    h_1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="valid")(input)
    
    h_2 = tf.keras.layers.Conv2D(filters=384,
                                 kernel_size=3,
                                 strides=2,
                                 padding="valid",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    h_3 = tf.keras.layers.Conv2D(filters=192,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    h_3 = tf.keras.layers.Conv2D(filters=224,
                                 kernel_size=3,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_3)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    h_3 = tf.keras.layers.Conv2D(filters=256,
                                 kernel_size=3,
                                 strides=2,
                                 padding="valid",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_3)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    h = tf.concat([h_1, h_2, h_3], 3)   # 17 x 17 x 1024

    return h

def reduction_B(input, weight_decay):

    h_1 = tf.keras.layers.MaxPool2D(pool_size=(3,3), strides=2, padding="valid")(input)

    h_2 = tf.keras.layers.Conv2D(filters=192,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    h_2 = tf.keras.layers.Conv2D(filters=192,
                                 kernel_size=3,
                                 strides=2,
                                 padding="valid",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_2)
    h_2 = tf.keras.layers.BatchNormalization()(h_2)
    h_2 = tf.keras.layers.ReLU()(h_2)

    h_3 = tf.keras.layers.Conv2D(filters=256,
                                 kernel_size=1,
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(input)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    h_3 = tf.keras.layers.Conv2D(filters=256,
                                 kernel_size=(1,7),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_3)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    h_3 = tf.keras.layers.Conv2D(filters=256,
                                 kernel_size=(7,1),
                                 strides=1,
                                 padding="same",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_3)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)

    h_3 = tf.keras.layers.Conv2D(filters=320,
                                 kernel_size=3,
                                 strides=2,
                                 padding="valid",
                                 use_bias=False,
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(h_3)
    h_3 = tf.keras.layers.BatchNormalization()(h_3)
    h_3 = tf.keras.layers.ReLU()(h_3)
    
    h = tf.concat([h_1, h_2, h_3], 3)

    return h

def inception_v4(input_shape=(299, 299, 3), weight_decay=0.00005):

    h = inputs = tf.keras.Input(input_shape)

    h = stem(h, weight_decay)   # 35 x 35 x 384

    for _ in range(4):
        h = inception_A(h, weight_decay)    # 35 x 35 x 384

    h = reduction_A(h, weight_decay)    # 17 x 17 x 1024
    
    for _ in range(7):
        h = inception_B(h, weight_decay)    # 17 x 17 x 1024

    h = reduction_B(h, weight_decay)    # 8 x 8 x 1536

    for _ in range(3):
        h = inception_C(h, weight_decay)    # 8 x 8 x 1536

    h = tf.keras.layers.GlobalAveragePooling2D()(h) # 1536
    h = tf.keras.layers.Dropout(0.2)(h)
    h = tf.keras.layers.Dense(1)(h)

    return tf.keras.Model(inputs=inputs, outputs=h)

model = inception_v4()
model.summary()