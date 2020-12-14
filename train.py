# -*- coding:utf-8 -*-
from se_resnet_model import *
from absl import flags
from random import random, shuffle

import numpy as np
import os
import sys

flags.DEFINE_string("tr_txt_path", "", "Training text path")

flags.DEFINE_string("tr_img_path", "", "Training image path")

flags.DEFINE_string("te_txt_path", "", "Training text path")

flags.DEFINE_string("te_img_path", "", "Training image path")

flags.DEFINE_integer("load_size", 256, "se-resnet:256, ...")

flags.DEFINE_integer("img_size", 224, "se-resnet:224, ...")

flags.DEFINE_integer("img_ch", 3, "Image's channels")

flags.DEFINE_integer("batch_size", 32, "Training batch size")

flags.DEFINE_float("lr", 0.001, "Learning rate")

flags.DEFINE_integer("epochs", 200, "Total epochs")

flags.DEFINE_bool("train", True, "True or False")

flags.DEFINE_bool("pre_checkpoint", False, "True or False")

flags.DEFINE_string("save_checkpoint", "", "Save checkpoint path")

flags.DEFINE_string("graphs", "", "")

# Need to write the test part

FLAGS = flags.FLAGS
FLAGS(sys.argv)

optim = tf.keras.optimizers.Adam(FLAGS.lr)
# It's Not Just Black and White: Classifying Defendant Mugshots Based on the Multidimensionality of Race and Ethnicity
def train_func(img_list, lab_list):

    img = tf.io.read_file(img_list)
    img = tf.image.decode_jpeg(img, FLAGS.img_ch)
    img = tf.image.resize(img, [FLAGS.load_size, FLAGS.load_size])
    img = tf.image.random_crop(img, [FLAGS.img_size, FLAGS.img_size, FLAGS.img_ch])

    if random > 0.5:
        img = tf.image.flip_left_right(img)
        img = tf.image.per_image_standardization(img)
    else:
        img = tf.image.per_image_standardization(img)

    lab = lab_list

    return img, lab

def test_func(img_path, lab_list):

    img = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img, FLAGS.img_ch)
    # img = tf.image.rgb_to_grayscale(img)
    img = tf.image.resize(img, [FLAGS.img_size, FLAGS.img_size])
    img = tf.image.per_image_standardization(img)

    lab = tf.cast(lab_list, tf.float32)
    return img, lab

@tf.function
def run_model(model, images, training=True):
    return model(images, training=training)

def cal_loss(model, batch_images, batch_labels):

    with tf.GradientTape() as tape:
        logits = run_model(model, batch_images, True)
        loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(batch_labels, logits)

    grads = tape.gradient(loss, model.trainable_variables)
    optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss

def cal_acc(model, images, labels):

    logits = run_model(model, images, False)
    logits = tf.nn.sigmoid(logits)  # [batch, 1]
    logits = tf.squeeze(logits, 1)  # [batch]

    predict = tf.cast(tf.greater(logits, 0.5), tf.float32)
    count_acc = tf.cast(tf.equal(predict, labels), tf.float32)
    count_acc = tf.reduce_sum(count_acc)

    return count_acc

def main():
    model = se_resnet_50()
    model.summary()

    if FLAGS.pre_checkpoint:
        ckpt = tf.train.Checkpoint()
        ckpt_manger = tf.train.CheckpointManager(ckpt, FLAGS.save_checkpoint, 5)

        if ckpt_manger.latest_checkpoint:
            ckpt.restore(ckpt_manger.latest_checkpoint)
            print("=====================")
            print("Restored!!!!!!")
            print("=====================")

    if FLAGS.train:
        count = 0
        tr_img = np.loadtxt(FLAGS.tr_txt_path, dtype="<U100", skiprows=0, usecols=0)
        tr_img = [FLAGS.tr_img_path + img for img in tr_img]
        tr_lab = np.loadtxt(FLAGS.tr_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_img = np.loadtxt(FLAGS.te_txt_path, dtype="<U100", skiprows=0 ,usecols=0)
        te_img = [FLAGS.te_img_path + img for img in te_img]
        te_lab = np.loadtxt(FLAGS.te_txt_path, dtype=np.int32, skiprows=0, usecols=1)

        te_gener = tf.data.Dataset.from_tensor_slices((te_img, te_lab))
        te_gener = te_gener.shuffle(len(te_img))
        te_gener = te_gener.map(test_func)
        te_gener = te_gener.batch(FLAGS.batch_size)
        te_gener = te_gener.prefetch(tf.data.experimental.AUTOTUNE)

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = FLAGS.graphs + current_time + '/train'
        val_log_dir = FLAGS.graphs + current_time + '/val'
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        val_summary_writer = tf.summary.create_file_writer(val_log_dir)

        for epoch in range(FLAGS.epochs):
            TR = list(zip(tr_img, tr_lab))
            shuffle(TR)
            tr_img, tr_lab = zip(*TR)

            tr_gener = tf.data.Dataset.from_tensor_slices((tr_img, tr_lab))
            tr_gener = tr_gener.shuffle(len(tr_img))
            tr_gener = tr_gener.map(train_func)
            tr_gener = tr_gener.batch(FLAGS.batch_size)
            tr_gener = tr_gener.prefetch(tf.data.experimental.AUTOTUNE)

            tr_idx = len(tr_img) // FLAGS.batch_size
            tr_iter = iter(tr_gener)
            for step in range(tr_idx):
                batch_images, batch_labels = next(tr_iter)

                loss = cal_loss(model, batch_images, batch_labels)  # Tomorrow!!!! Remember this!!!
                
                if count % 10 == 0:
                    print("Epoch: {} [{}/{}] total loss = {}".format(epoch, step + 1, tr_idx, loss))

                if count % 100 == 0:
                    # test
                    te_idx = len(te_img) // FLAGS.batch_size
                    te_iter = iter(te_gener)
                    count_acc = 0.
                    for i in range(te_idx):
                        te_images, te_labels = next(te_iter)
                        count_acc += cal_acc(ethnicity_MODEL, te_images, te_labels)

                    ACC = (count_acc / len(te_img)) * 100.
                    print("Acc = {} for {} steps".format(ACC, count))

                    with val_summary_writer.as_default():
                        tf.summary.scalar('ACC', ACC, step=count)


                if count % 500 == 0:
                    num_ = int(count // 500)
                    model_dir = "%s/%s" % (FLAGS.save_checkpoint, num_)

                    ckpt = tf.train.Checkpoint(ethnicity_MODEL=ethnicity_MODEL,
                                               optim=optim)
                    ckpt_dir = model_dir + "/" + "ethnic_model_{}.ckpt".format(count)

                    ckpt.save(ckpt_dir)

                count += 1

if __name__ == "__main__":
    main()