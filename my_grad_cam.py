import os
import numpy as np
import cv2
import tensorflow as tf
import standardmodel
import DenseNet

PLACES365_MEAN = [104.05100722, 112.51448911, 116.67603893]
IMAGENET_MEAN = [104.00698793, 116.66876762, 122.67891434]
# standard normalization (mean, std), RGB format
Image_Norm = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
tmp_dir = '/home/hyy/grad_cam/tmp'
wpath = '/home/hyy/MIT67-vgg16-places365-fc-SGD0.004_epoch19_0.8187.ckpt'

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

def mit67_sdict():
    category_path = '/home/0_public_data/MIT67/ClassNames.txt'
    sdict = {}
    with open(category_path, 'r') as fi:
        for sid, line in enumerate(fi):
            sname = line.strip()
            sdict[sid] = sname
    return sdict
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
    weights = tf.reduce_mean(grads, axis=(0, 1))
    cam = tf.reduce_sum(weights * output, axis=-1)
    return cam, logits[0]

def visual_saliency():
    imdir = '/home/0_public_data/MIT67/Images'
    destdir = os.path.join(tmp_dir, 'Visualization/MIT67')
    if not os.path.exists(destdir):
        os.makedirs(destdir)
    train_file = '/home/0_public_data/MIT67/TrainImages.label'
    train_list, train_labels, train_dpath = get_im_list(imdir, destdir, train_file)
    for i in range(0, train_list.__len__()):
        impath = train_list[i]
        im = cv2.imread(impath)
        im = cv2.resize(im, dsize=(224, 224))
        mean_im = np.tile(IMAGENET_MEAN, [224, 224, 1])
        x = im - mean_im
        nclass = 67
        model = standardmodel.VGG16(nclass)
        # model = DenseNet.DenseNet161(nclass)
        cate_id = tf.placeholder(tf.int32)
        tfcam, logits = grad_cam(model, x[None, :, :, :], cate_id, "conv5_3", nclass)
        score = tf.nn.softmax(logits)
        # Configuration of GPU usage
        config = tf.ConfigProto()
        # config.gpu_options.per_process_gpu_memory_fraction = 0.7
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as sess:
            if(i<=0):
                model.init_from_ckpt(wpath)
            # Initialize all variables
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            #model.load_initial_weights(sess, wpath)
            sco = sess.run(score)
            print(_sdict[np.argmax(sco)], np.max(sco))
            cid = np.argmax(sco)

            ocam = sess.run(tfcam, feed_dict={cate_id: cid})
            ocam = np.maximum(ocam, 0)
            ocam = cv2.resize(ocam, (224, 224))
            if np.mean(ocam) == 0:
                print(_sdict[cid])
            heatmap = ocam / np.max(ocam)
            gcam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
            gcam = np.float32(gcam) + np.float32(im)
            gcam = np.uint8(255 * gcam / np.max(gcam))
            cv2.imwrite(train_dpath[i].format(_sdict[cid]), gcam)



if __name__ == '__main__':
    visual_saliency()

