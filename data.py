import os
import numpy as np
import tensorflow as tf

AUG=False  #是否进行图像增强？
img_size=[96,96]
CHANNELS=1

def data_augmentation(images):
    images = tf.image.random_brightness(images, max_delta=0.3)
    images = tf.image.random_contrast(images, 0.8, 1.2)
    return images

def _parse_function(filename, label):
    image_decoded = tf.image.decode_png(tf.io.read_file(filename), channels=CHANNELS)
    image_decoded = tf.cast(image_decoded, tf.float32)
    image_decoded = (255. - image_decoded)
    image_decoded = tf.image.resize_with_crop_or_pad(image_decoded, img_size[0], img_size[1])

    if AUG:
        image_decoded = data_augmentation(image_decoded)
    label = tf.cast(label, tf.int32)
    return image_decoded, label

lable_dict = []

def load_data(path, buffer_size, batch_size, channels=1, aug=False):
    AUG=aug
    CHANNELS=channels
    file_and_label = []

    dirs = os.listdir(path)
#    dirs.sort()

    i = 0
    for label_name in dirs:
        lable_dict.append(label_name)
        for file_name in os.listdir(path + '/' + label_name):
            file_and_label.append([i, path + '/' + label_name + '/' + file_name])

        i = i + 1

    file_and_label = np.array(file_and_label)
    np.random.shuffle(file_and_label)
    labels = list(map(int,file_and_label[:,0]))
    files = list(file_and_label[:, 1])

    files = tf.constant(files)
    labels = tf.constant(labels)

    dataset = tf.data.Dataset.from_tensor_slices((files, labels))
    dataset = dataset.map(_parse_function)
    if (buffer_size==0): #testing
        dataset = dataset.batch(batch_size)
    else:
        dataset = dataset.shuffle(buffer_size=buffer_size).batch(batch_size).repeat()

    image_batch, label_batch = dataset.make_one_shot_iterator().get_next()

    return label_batch, image_batch
