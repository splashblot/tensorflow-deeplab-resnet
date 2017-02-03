"""Training script for the DeepLab-ResNet network on the PASCAL VOC dataset
   for semantic image segmentation.

This script fine-tunes the model using augmented PASCAL VOC,
which contains approximately 10000 images for training and 1500 images for validation.
Only the last 'fc1_voc12' layers are being trained.
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

# Choose solver: 1:SGD, 2:ADAM
SOLVER_MODE = 1  
# SOLVER_MODE = 2

# CamVid
#N_CLASSES = 12
#ignore_label = 11
#DATA_DIRECTORY = '/home/garbade/datasets/CamVid/'
#DATA_LIST_PATH = '/home/garbade/models/03_CamVid/02_DL_v2_CamVid_ResNet_CRF/camvid/list/train.txt'
#TRAIN_SIZE = 400

# Voc12
N_CLASSES = 21
DATA_DIRECTORY = '/home/garbade/datasets/VOC2012/'
DATA_LIST_PATH = './dataset/train.txt'
TRAIN_SIZE = 10000


BATCH_SIZE = 4
INPUT_SIZE = '321,321'
LEARNING_RATE = 1e-4
LEARNING_RATE_GIST = 1e-6
NUM_STEPS = 20000
# RESTORE_FROM = './deeplab_tf_model/deeplab_resnet.ckpt'
RESTORE_FROM = './deeplab_tf_model/deeplab_resnet_init.ckpt'
SAVE_DIR = './images_finetune/'
SAVE_NUM_IMAGES = 2
SAVE_PRED_EVERY = 100
SNAPSHOT_DIR = './snapshots_finetune_sgd/'

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
                        help="Learning rate for training last layer.")
    parser.add_argument("--learning-rate-gist", type=float, default=LEARNING_RATE_GIST,
                        help="Learning rate for everything but last layer.")
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
    parser.add_argument("--save_dir", type=str, default=SAVE_DIR,
                        help="Where to save figures with predictions.")
    parser.add_argument("--train_size", type=int, default=TRAIN_SIZE,
                        help="Number of train images (to drive learning rate decay).")                          
    parser.add_argument("--n_classes", type=int, default=N_CLASSES,
                        help="Number of classes.")
    parser.add_argument("--solver", type=int, default=SOLVER_MODE,
                        help="Solver mode: 1: SGD, 2: Adam.")
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
            args.random_scale,
            coord)
        image_batch, label_batch = reader.dequeue(args.batch_size)
    
    # Create network.
    net = DeepLabResNetModel({'data': image_batch}, args.n_classes, is_training=args.is_training)
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
    vars_all = tf.global_variables()
    vars_restore_gist = [v for v in tf.global_variables() if not 'fc1_voc12' in v.name] # Restore everything but last layer
    vars_restore_last_layer = [v for v in tf.global_variables() if 'fc1_voc12' in v.name] # Restore everything but last layer
    vars_trainable_gist = [v for v in vars_restore_gist if 'weights' in v.name] # MG
    vars_trainable_last_layer = vars_restore_last_layer 
    
    prediction = tf.reshape(raw_output, [-1, args.n_classes]) # Dim = [6724,12] --> 6724 = 41 x 41 x 4 = H x W x N
    label_proc = prepare_label(label_batch, tf.pack(raw_output.get_shape()[1:3]),args.n_classes)
    gt = tf.reshape(label_proc, [-1, args.n_classes]) # Dim = [6724,12]

    # Pixel-wise softmax loss.
    loss = tf.nn.softmax_cross_entropy_with_logits(prediction, gt) # Dim = [6724,1]
    reduced_loss = tf.reduce_mean(loss)
#    tf.summary.histogram("loss_vec", loss) # MG
    
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
    summary_writer = tf.summary.FileWriter(args.snapshot_dir)
   
    if SOLVER_MODE == 1:
        # SGD Optimization
        batch = tf.Variable(0)
        
        lr = tf.train.exponential_decay(
          args.learning_rate,  # Base learning rate.
          batch * BATCH_SIZE,  # Current index into the dataset.
          TRAIN_SIZE,          # Decay step.
          0.95,                # Decay rate.
          staircase=True)
        lr_gist = tf.train.exponential_decay(
          args.learning_rate_gist,   # Base learning rate.
          batch * BATCH_SIZE,        # Current index into the dataset.
          TRAIN_SIZE,                # Decay step.
          0.95,                      # Decay rate.
          staircase=True)
          
        # Use simple momentum for the optimization.
        optimizer = tf.train.MomentumOptimizer(lr,0.9)
        optim = optimizer.minimize(reduced_loss,global_step=batch,var_list=vars_trainable_gist)
        optimizer_gist = tf.train.MomentumOptimizer(lr_gist,0.9) 
        optim_gist = optimizer_gist.minimize(reduced_loss,global_step=batch,var_list=vars_trainable_gist)
    elif SOLVER_MODE == 2:
        # Define loss and optimisation parameters. Slow optimization for everything but the last layer
        optimizer = tf.train.AdamOptimizer(learning_rate=args.learning_rate)
        optim = optimizer.minimize(reduced_loss, var_list=vars_trainable_last_layer)
        optimizer_gist = tf.train.AdamOptimizer(learning_rate=args.learning_rate_gist)
        optim_gist = optimizer_gist.minimize(reduced_loss,var_list=vars_trainable_gist)
    else:
        print('Error: Choose solver mode!')
        
    
    # Set up tf session and initialize variables. 
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()
    # Log variables
#    summary_writer = tf.summary.FileWriter("./logs/nn_logs", sess.graph) # MG
#    tf.summary.scalar("reduced_loss", reduced_loss) # MG
#    var = [v for v in tf.global_variables() if v.name.startswith('res5c_branch2c/weights')][0]
#    tf.summary.histogram("bn5c_branch2c/mean_0", var) # MG
#    var = [v for v in tf.trainable_variables() if v.name == "fc1_voc12_c0/weights/read:0"][0]
#    tf.summary.histogram("fc1_voc12_c0/weights_0", var) # MG
#    var = [v for v in tf.trainable_variables() if v.name == "bn5c_branch2c/moving_mean/read:0"][0]
#    tf.summary.histogram("bn5c_branch2c/BatchNorm/beta_0", var) # MG
#    merged_summary_op = tf.summary.merge_all() # MG
    
    sess.run(init)
    
    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=vars_all, max_to_keep=1)
    
    # Load variables if the checkpoint is provided.
    if args.restore_from is not None:
        # loader = tf.train.Saver(var_list=restore_var)
        loader = tf.train.Saver(var_list=vars_restore_gist) # MG: restore everything but last layer
        load(loader, sess, args.restore_from)
    
    # Start queue threads.
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    # Iterate over training steps.
    losses = []
    for step in range(args.num_steps):
        start_time = time.time()
        
        if step % args.save_pred_every == 0:
            # loss_value, images, labels, preds, summary, _ = sess.run([reduced_loss, image_batch, label_batch, pred, total_summary, optim])
            loss_value, images, labels, preds, summary, _, _ = sess.run([reduced_loss, image_batch, label_batch, pred, total_summary, optim_gist,optim])
            losses.append(loss)
            average_loss =  sum(losses)/len(losses)
            summary_writer.add_summary(summary, step)
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
            save(saver, sess, args.snapshot_dir, step)
        else:
            # loss_value, _ = sess.run([reduced_loss, optim])
            loss_value, _, _= sess.run([reduced_loss, optim_gist, optim])
            losses.append(loss)
            
        duration = time.time() - start_time
        print('step {:d} \t loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))
    coord.request_stop()
    coord.join(threads)
    
if __name__ == '__main__':
    main()
