"""Evaluation script for the DeepLab-ResNet network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on 1449 validation images.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, prepare_label


OUTPUT_IMGS = False

# CamVid
#n_classes = 12
#DATA_DIRECTORY = '/home/garbade/datasets/CamVid/'
#DATA_LIST_PATH = '/home/garbade/datasets/CamVid/list/test_70.txt'
#DATA_LIST_PATH_ID = '/home/garbade/datasets/CamVid/list/test_id.txt'
#RESTORE_FROM = './snapshots_finetune/2017-01-31-CamVid_Loss-0-2/model.ckpt-4400'
#SAVE_DIR = './images_val/voc12/'

# Voc12
n_classes = 21
DATA_DIRECTORY = '/home/garbade/datasets/VOC2012/'
DATA_LIST_PATH = './dataset/val.txt'
DATA_LIST_PATH_ID = '/home/garbade/models/01_voc12/17_DL_v2_ResNet/voc12/list/val_id.txt'
RESTORE_FROM = '/home/garbade/models_tf/01_voc12/02_finetune_adam/snapshots_finetune/model.ckpt-19900'
# RESTORE_FROM = './snapshots_finetune/model.ckpt-1400'
# RESTORE_FROM = './deeplab_tf_model/deeplab_resnet.ckpt'
SAVE_DIR = './images_val/voc12/'


#NUM_STEPS = 1449 # Number of images in the validation set.


imgList = []
with open(DATA_LIST_PATH_ID, "rb") as fp:
    for i in fp.readlines():
        tmp = i[:-1]
        try:
            imgList.append(tmp)
        except:pass

if imgList == []:
    print('Error: Filelist is empty')
else:
    print('Filelist loaded successfully')
NUM_STEPS = len(imgList)
print(NUM_STEPS)
def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLabLFOV Network")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of images in the validation set.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--n_classes", type=int, default=n_classes,
                        help="How many classes to predict (default = 21).")
    return parser.parse_args()

def load(saver, sess, ckpt_path):
    '''Load trained weights.
    
    Args:
      saver: TensorFlow saver object.
      sess: TensorFlow session.
      ckpt_path: path to checkpoint file with parameters.
    ''' 
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))

def main():
    """Create the model and start the evaluation process."""
    args = get_arguments()
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            None, # No defined input size.
            False, # No random scale.
            coord)
        image, label = reader.image, reader.label
    image_batch, label_batch = tf.expand_dims(image, dim=0), tf.expand_dims(label, dim=0) # Add one batch dimension.

    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, args.n_classes, is_training=False)

    # Which variables to load.
    restore_var = tf.global_variables()
    
    # Predictions.
    raw_output = net.layers['fc1_voc12']
    raw_output = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output = tf.argmax(raw_output, dimension=3)
    pred = tf.expand_dims(raw_output, dim=3) # Create 4-d tensor.
    
    # mIoU
    pred = tf.reshape(pred, [-1,])
    gt = tf.reshape(label_batch, [-1,])
    weights = tf.cast(tf.less_equal(gt, 20), tf.int32) # Ignore void label '255'.
    mIoU, update_op = tf.contrib.metrics.streaming_mean_iou(pred, gt, num_classes=21, weights=weights)
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    
    # Load weights.
    loader = tf.train.Saver(var_list=restore_var)
    if args.restore_from is not None:
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)
    
    # Iterate over training steps.
    for step in range(args.num_steps):
        preds, _ = sess.run([pred, update_op])
        if step % 100 == 0:
            print('step {:d}'.format(step))
        if OUTPUT_IMGS:
            # print(np.array(preds).shape)
            msk = decode_labels(np.array(preds)[0, :, :, 0], args.n_classes)
            im = Image.fromarray(msk)
            im.save(args.save_dir + imgList[step] + '.png')
            print('File saved to {}'.format(args.save_dir + imgList[step] + '.png'))
    print('Mean IoU: {:.3f}'.format(mIoU.eval(session=sess)))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
