# -*- coding: utf-8 -*-
"""
Created on 6/11/18

Description: visualize saliency region by Grad-CAM method

@author: Gongwei Chen
"""


import os
import numpy as np
import cv2

import tensorflow as tf

from projects.Visualization import standardmodel
from projects.Visualization import DenseNet


# mean, BGR format
PLACES365_MEAN = [104.05100722, 112.51448911, 116.67603893]
IMAGENET_MEAN = [104.00698793, 116.66876762, 122.67891434]
# standard normalization (mean, std), RGB format
Image_Norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tmp_dir = '/home/cgw/Data2-CGW/tmp/TF'
wpath = '/home/cgw/Data2-CGW/tmp/TF/models/pre-trained_models/vgg16-places365.npy'
# wpath = '/home/cgw/Data2-CGW/tmp/TF/models/pre-trained_models/vgg16-ImageNet.npy'
# wpath = '/home/cgw/Data2-CGW/tmp/TF/models/pre-trained_models/densenet161_places365_pytorch.npy'
# wpath = '/home/cgw/Data2-CGW/tmp/TF/MIT67_data/finetune_vgg16/MIT67-vgg16-places365-fc-SGD0.004_epoch19_0.8187.ckpt'


def get_im_list(im_dir, dest_dir, file_path):
    im_list = []
    im_labels = []
    dest_path = []
    with open(file_path, 'r') as fi:
        for line in fi:
            im_list.append(os.path.join(im_dir, line.split()[0]))
            im_labels.append(int(line.split()[-1]))
            fnewname = '_'.join(line.split()[0][:-4].split('/'))
            dest_path.append(os.path.join(dest_dir, fnewname + '_{}.jpg'))

    return im_list, im_labels, dest_path


def p365_sdict():
    category_path = '/home/cgw/Data2-CGW/Datasets/Places/Places365/categories_places365.txt'

    sdict = {}
    with open(category_path, 'r') as fi:
        for line in fi:
            sname = line.split()[0][3:]
            sid = int(line.split()[-1])
            sname = '_'.join(sname.split('/'))
            sdict[sid] = sname

    return sdict


def mit67_sdict():
    category_path = '/home/cgw/Data2-CGW/Datasets/MIT67/ClassNames.txt'

    sdict = {}
    with open(category_path, 'r') as fi:
        for sid, line in enumerate(fi):
            sname = line.strip()
            sdict[sid] = sname

    return sdict

# _sdict = p365_sdict()
_sdict = mit67_sdict()


def grad_cam(input_model, x, category_index, layer_name, nb_class):

    x = tf.convert_to_tensor(x, dtype=tf.float32)
    model_out = input_model.inference(x)
    outlayer = model_out[layer_name]
    logits = model_out['logits']
    loss = logits * tf.one_hot(category_index, nb_class, dtype=tf.float32)

    grad_var = tf.gradients(loss, outlayer)[0]
    output, grads = outlayer[0], grad_var[0]
    grads = grads / tf.norm(grads)
    # grads = grads / (tf.sqrt(tf.reduce_mean(tf.square(grads))) + 1e-5)

    weights = tf.reduce_mean(grads, axis=(0, 1), keepdims=True)
    cam = tf.reduce_sum(weights * output, axis=-1)

    return cam, logits[0]


def visual_saliency():

    imdir = '/home/cgw/Data2-CGW/Datasets/MIT67/Images'
    destdir = os.path.join(tmp_dir, 'Visualization/MIT67/VGG16_ft/conv4')
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    train_file = '/home/cgw/Data2-CGW/Datasets/MIT67/TrainImages.label'
    test_file = '/home/cgw/Data2-CGW/Datasets/MIT67/TestImages.label'

    train_list, train_labels, train_dpath = get_im_list(imdir, destdir, train_file)

    impath = train_list[19]
    # impath = '/home/cgw/Desktop/cat_dog.png'
    im = cv2.imread(impath)  # BGR order
    im = cv2.resize(im, dsize=(224, 224))
    # for mean normalization, VGG
    mean_im = np.tile(IMAGENET_MEAN, [224, 224, 1])
    x = im - mean_im
    # for standard normalization, DenseNet
    # x = im[:, :, ::-1]  # RGB order
    # mean_im = np.tile(Image_Norm[0], [224, 224, 1])
    # x = x / 255.0 - mean_im
    # x /= Image_Norm[1]

    nclass = 67
    model = standardmodel.VGG16(nclass)
    # model = DenseNet.DenseNet161(nclass)
    cate_id = tf.placeholder(tf.int32)

    tfcam, logits = grad_cam(model, x[None, :, :, :], cate_id, "conv5_3", nclass)
    # tfcam, logits = grad_cam(model, x[None, :, :, :], cate_id, "norm5", nclass)
    score = tf.nn.softmax(logits)

    # Configuration of GPU usage
    config = tf.ConfigProto()
    # config.gpu_options.per_process_gpu_memory_fraction = 0.7
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        model.init_from_ckpt(wpath)
        # Initialize all variables
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())

        # model.load_initial_weights(sess, wpath)

        sco = sess.run(score)
        print(_sdict[np.argmax(sco)], np.max(sco))

        for cid in range(nclass):
            ocam = sess.run(tfcam, feed_dict={cate_id: cid})

            ocam = np.maximum(ocam, 0)
            ocam = cv2.resize(ocam, (224, 224))
            if np.mean(ocam) == 0:
                print(_sdict[cid])
            heatmap = ocam / np.max(ocam)
            gcam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            gcam = np.float32(gcam) + np.float32(im)
            gcam = np.uint8(255 * gcam / np.max(gcam))
            cv2.imwrite(train_dpath[19].format(_sdict[cid]), gcam)
            # cv2.imwrite(train_dpath[0].format(cid), gcam)
            # cv2.imwrite('/home/cgw/Desktop/cat_dog_{}.jpg'.format(cid), gcam)


if __name__ == '__main__':
    visual_saliency()

