"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script trains the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
"""

from __future__ import print_function

import argparse
from datetime import datetime
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from deeplab_resnet import DeepLabResNetModel, ImageReader, decode_labels, decode_labels_old, inv_preprocess, prepare_label


SOLVER_MODE = 1

#### CamVid
DATASET = 'CAMVID'
n_classes = 11
ignore_label = 10
DATA_DIRECTORY = '/home/garbade/datasets/CamVid/'
DATA_LIST_PATH = './dataset/camvid/train.txt'
OUTPUT_ROOT = '/home/garbade/models_tf/03_CamVid/16_fixedRandomCropping/'
#RESTORE_FROM = './deeplab_tf_model/deeplab_resnet_init.ckpt'


#### Cityscapes (19 classes + BG)
# DATASET = 'CITY'
#n_classes=19
#ignore_label=18
#DATA_DIRECTORY='/home/garbade/datasets/cityscapes/'
#DATA_LIST_PATH='./dataset/city/small_50/train_aug.txt'
#OUTPUT_ROOT='/home/garbade/models_tf/05_Cityscapes/10_fixedMirrorImgAndScale/'
#RESTORE_FROM = './deeplab_tf_model/deeplab_resnet_init.ckpt'

#### voc12
# DATASET = 'VOC2012'
#n_classes = 21
#ignore_label = 20
#DATA_DIRECTORY = '/home/garbade/datasets/VOC2012/'
#DATA_LIST_PATH = './dataset/voc12/train.txt'
#OUTPUT_ROOT='/home/garbade/models_tf/01_voc12/10_fixedMirrorImgAndScale/'
#RESTORE_FROM = './deeplab_tf_model/deeplab_resnet_init.ckpt'



BATCH_SIZE = 10
INPUT_SIZE = '321,321'
LEARNING_RATE = 2.5e-4
#LEARNING_RATE = 2.5e-3
NUM_STEPS = 20001

SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 100
SAVE_DIR = OUTPUT_ROOT + '/images_finetune/'
SNAPSHOT_DIR = OUTPUT_ROOT + '/snapshots_finetune/'
LOG_DIR = OUTPUT_ROOT + '/logs/'


print('Dataset: ' + DATASET + '\n' + 
          'Restore from: ' + RESTORE_FROM)

## OPTIMISATION PARAMS ##
WEIGHT_DECAY = 0.0005
# BASE_LR = 2.5e-4
BASE_LR = LEARNING_RATE
POWER = 0.9
MOMENTUM = 0.9
## OPTIMISATION PARAMS ##

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)


def get_arguments():
    """Parse all the arguments provided from the CLI.
    
    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Learning rate for training.")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-num-images", type=int, default=SAVE_NUM_IMAGES,
                        help="How many images to save.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--log_dir", type=str, default=LOG_DIR,
                        help="Where to save logs of the model.")                        
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save figures with predictions.")
    parser.add_argument("--n_classes", type=int, default=n_classes,
                        help="Number of classes.")
    parser.add_argument("--ignore_label", type=int, default=ignore_label,
                        help="Number of classes.")
    return parser.parse_args()

def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print('The checkpoint has been created.')

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
    """Create the model and start the training."""
    args = get_arguments()
    
    h, w = map(int, args.input_size.split(','))
    input_size = (h, w)
    
    # Create queue coordinator.
    coord = tf.train.Coordinator()
    
    # Load reader.
    with tf.name_scope("create_inputs"):
        reader = ImageReader(
            args.data_dir,
            args.data_list,
            input_size,
            'train',    # phase is either 'train', 'val' or 'test'
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
    
    # Create network.
    net = DeepLabResNetModel({'data': image_batch},args.n_classes, is_training=args.is_training)
    # For a small batch size, it is better to keep 
    # the statistics of the BN layers (running means and variances)
    # frozen, and to not update the values provided by the pre-trained model. 
    # If is_training=True, the statistics will be updated during the training.
    # Note that is_training=False still updates BN parameters gamma (scale) and beta (offset)
    # if they are presented in var_list of the optimiser definition.

    # Predictions.
    raw_output = net.layers['fc1_voc12']
    # Which variables to load. Running means and variances are not trainable,
    # thus all_variables() should be restored.
    restore_var = tf.global_variables()
    all_trainable = [v for v in tf.trainable_variables() if 'beta' not in v.name and 'gamma' not in v.name]
    fc_trainable = [v for v in all_trainable if 'fc' in v.name]
    conv_trainable = [v for v in all_trainable if 'fc' not in v.name] # lr * 1.0
    fc_w_trainable = [v for v in fc_trainable if 'weights' in v.name] # lr * 10.0
    fc_b_trainable = [v for v in fc_trainable if 'biases' in v.name] # lr * 20.0
    assert(len(all_trainable) == len(fc_trainable) + len(conv_trainable))
    assert(len(fc_trainable) == len(fc_w_trainable) + len(fc_b_trainable))
    
    vars_restore_gist = [v for v in tf.global_variables() if not 'fc' in v.name] # Restore everything but last layer
    
    ## TODO: Here everything below n_classes is being ignored  -> match this with ingnoer_label = 255 -> IGNORE 255 ##
    raw_prediction = tf.reshape(raw_output, [-1, args.n_classes])
    label_proc = prepare_label(label_batch, tf.pack(raw_output.get_shape()[1:3]),args.n_classes, one_hot=False) # [batch_size, h, w]
    raw_gt = tf.reshape(label_proc, [-1,])
    indices = tf.squeeze(tf.where(tf.less_equal(raw_gt, args.n_classes - 1)), 1)
    gt = tf.cast(tf.gather(raw_gt, indices), tf.int32)
    prediction = tf.gather(raw_prediction, indices)
                                                  
                                                  
    # Pixel-wise softmax loss.
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=gt)
    l2_losses = [WEIGHT_DECAY * tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'weights' in v.name]
    reduced_loss = tf.reduce_mean(loss) + tf.add_n(l2_losses)
    
    # Processed predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, tf.shape(image_batch)[1:3,])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = tf.expand_dims(raw_output_up, dim=3)
    
    # Image summary.
    images_summary = tf.py_func(inv_preprocess, [image_batch, args.save_num_images], tf.uint8)
    labels_summary = tf.py_func(decode_labels, [label_batch, args.save_num_images], tf.uint8)
    preds_summary = tf.py_func(decode_labels, [pred, args.save_num_images], tf.uint8)
    
    total_summary = tf.summary.image('images', 
                                     tf.concat(0, [images_summary, labels_summary, preds_summary]), 
                                     max_outputs=args.save_num_images) # Concatenate row-wise.
                                     #tf.concat(2, [images_summary, labels_summary, preds_summary]), 
                                     #max_outputs=args.save_num_images) # Concatenate row-wise.
    # summary_writer = tf.summary.FileWriter(args.log_dir)
   
    # Define loss and optimisation parameters.
    
    ## OPTIMISER ##
    base_lr = tf.constant(BASE_LR)
    step_ph = tf.placeholder(dtype=tf.float32, shape=())
    learning_rate = tf.scalar_mul(base_lr, tf.pow((1 - step_ph / 20000), POWER))
    tf.summary.scalar('learning_rate', learning_rate)

    if SOLVER_MODE == 1:
        opt_conv = tf.train.MomentumOptimizer(learning_rate, MOMENTUM)
        opt_fc_w = tf.train.MomentumOptimizer(learning_rate * 10.0, MOMENTUM)
        opt_fc_b = tf.train.MomentumOptimizer(learning_rate * 20.0, MOMENTUM)    
    elif SOLVER_MODE == 2:
        opt_conv = tf.train.AdamOptimizer(learning_rate=BASE_LR)
        opt_fc_w = tf.train.AdamOptimizer(learning_rate=BASE_LR * 10.0)
        opt_fc_b = tf.train.AdamOptimizer(learning_rate=BASE_LR * 20.0)
    else:
        print('Error: No SOLVER_MODE specified')
    

    grads = tf.gradients(reduced_loss, conv_trainable + fc_w_trainable + fc_b_trainable)
    grads_conv = grads[:len(conv_trainable)]
    grads_fc_w = grads[len(conv_trainable) : (len(conv_trainable) + len(fc_w_trainable))]
    grads_fc_b = grads[(len(conv_trainable) + len(fc_w_trainable)):]

    train_op_conv = opt_conv.apply_gradients(zip(grads_conv, conv_trainable))
    train_op_fc_w = opt_fc_w.apply_gradients(zip(grads_fc_w, fc_w_trainable))
    train_op_fc_b = opt_fc_b.apply_gradients(zip(grads_fc_b, fc_b_trainable))

    train_op = tf.group(train_op_conv, train_op_fc_w, train_op_fc_b)
    ## OPTIMISER ##
    
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    # Log variables
    summary_writer = tf.summary.FileWriter(args.log_dir, sess.graph) # MG
    tf.summary.scalar("reduced_loss", reduced_loss) # MG
    for v in conv_trainable + fc_w_trainable + fc_b_trainable: # Add histogram to all variables
        tf.summary.histogram(v.name.replace(":","_"),v)
    merged_summary_op = tf.summary.merge_all() # MG
    
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=restore_var, max_to_keep=1)
    
    # Load variables if the checkpoint is provided.
    if args.restore_from is not None:
        loader = tf.train.Saver(var_list=vars_restore_gist)
        #loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    # Create save_dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Iterate over training steps.
    for step in range(args.num_steps):
        start_time = time.time()
        feed_dict = { step_ph : step }
        
        if step % args.save_pred_every == 0:
            # loss_value, images, labels, preds, summary, _ = sess.run([reduced_loss, image_batch, label_batch, pred, total_summary, train_op], feed_dict=feed_dict) # total summary
            loss_value, images, labels, preds, summary, _ = sess.run([reduced_loss, image_batch, label_batch, pred, merged_summary_op, train_op], feed_dict=feed_dict)
            summary_writer.add_summary(summary, step)
            ### Print intermediary images
            fig, axes = plt.subplots(args.save_num_images, 3, figsize = (16, 12))
            for i in xrange(args.save_num_images):
                axes.flat[i * 3].set_title('data')
                axes.flat[i * 3].imshow((images[i] + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

                axes.flat[i * 3 + 1].set_title('mask')
                axes.flat[i * 3 + 1].imshow(decode_labels_old(labels[i, :, :, 0], args.n_classes))

                axes.flat[i * 3 + 2].set_title('pred')
                axes.flat[i * 3 + 2].imshow(decode_labels_old(preds[i, :, :, 0], args.n_classes))
            plt.savefig(args.save_dir + str(start_time) + ".png")
            plt.close(fig)
            ###
            save(saver, sess, args.snapshot_dir, step)
        else:
            loss_value, _ = sess.run([reduced_loss, train_op], feed_dict=feed_dict)
        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
