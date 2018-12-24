import os
import argparse

parser = argparse.ArgumentParser(description="Training networks on RECommended dataset.",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("-l", "--level", type=int, choices=[1, 2, 3], default=3,
                    help="Select how many hierarchical levels to utilize.\n"
                         "-l 1/--level 1: 1-Hierarchy.\n"
                         "-l 1/--level 2: 2-Hierarchy.\n"
                         "-l 3/--level 3: 3-Hierarchy (default).\n")
parser.add_argument("-f", "--fold", type=str, metavar='N', default='123',
                    help="Perform training-validation on specified folds.\n"
                         "-f 1/--fold 1: use fold 1 as validation set and folds 2,3 as training sets.\n"
                         "-f 123/--fold 123: 3-fold cross validation (default).\n")
parser.add_argument("-e", "--epoch", type=int, metavar='N', default=500,
                    help="-e N/--epoch N: determine how many epochs to train. The default is 500.\n"
                         "Early-stopping will be used, so feel free to increase it.")
parser.add_argument("-b", "--batch", type=int, metavar='N', default=64,
                    help="-b N/--batch N: manually set the batch size to N. The default is 64.\n"
                         "Current training session DOES NOT support batch size smaller than 32.")
args = parser.parse_args()
if args.batch < 32:
    raise argparse.ArgumentTypeError("Minimum batch size is 32")

import numpy as np
import matplotlib.pyplot as plt
from timeit import default_timer as timer
import tensorflow as tf
import keras.backend as K
from keras import applications
from keras.preprocessing.image import img_to_array, load_img
from keras.optimizers import Nadam

from utils import angular_error, percentile_mean
from config import *
from model import model_builder

# load configuration based on the pre-trained dataset
dataset_config = get_dataset_config(dataset='R')
# network architecture selection
model_config = get_model_config(level=args.level, confidence=False)

# configurations
##############################
DATASET = dataset_config['dataset']
PATCHES = dataset_config['patches']  # the number of square sub-images
PATCH_SIZE = dataset_config['patch_size']  # the size of square sub-image
NETWORK = model_config['network']
LR = model_config['LR']
BATCH_SIZE = args.batch
EPOCHS = args.epoch
FOLDS = [int(f) for f in args.fold]
PATIENCE = 6
MIN_DELTA = 0.01
FACTOR = 0.9
MIN_LR = LR/10.
MAX_LR = LR
EARLY_STOP_PATIENCE = 150
EPSILON = 1E-9
##############################


def get_train_batch():
    """
    generate image batch and labels iteratively
    Note: we highly recommend using Keras Sequence class to create a data generator, which would be ~1.5x faster than
    using this data generation function. However, Keras's 'fit_generator' method does not support callbacks for
    validation data, to which end we have to use 'predict_on_batch' method and manually evaluate the accuracies.
    :return: image batch as Numpy array, labels as Numpy array, and a bool indicator for continuing generating or stop
    """
    global current_train_index
    global continue_train
    local_index = 0
    img_batch = np.zeros(shape=(BATCH_SIZE, *PATCH_SIZE, 3))
    label_batch = np.zeros(shape=(BATCH_SIZE, 3))
    while local_index < BATCH_SIZE and continue_train:
        img_ID = train_img_IDs[current_train_index]
        img_batch[local_index] = img_to_array(load_img(img_ID))/255.
        label_batch[local_index] = train_labels[img_ID]
        local_index += 1
        current_train_index += 1
    if current_train_index+BATCH_SIZE >= len(train_img_IDs):
        continue_train = False
    return img_batch, label_batch, continue_train


def get_val_batch():
    """
    generate image batch and labels iteratively
    Note: the batch size for the validation set is different from BATCH_SIZE in the training phase. We collect all
    sub-images from ONE full-resolution image into a batch when evaluating on the validation set. The number of
    sub-images for an arbitrary full-resolution image need to be determined dynamically.
    :return: image batch as Numpy array, labels as Numpy array, and a bool indicator for continuing generating or stop
    """
    global current_val_index
    global continue_val
    local_index = 0
    val_batch_size = 1
    current_index = current_val_index
    while val_source_img_IDs[current_index+1] == val_source_img_IDs[current_index] and current_index+1 < len(val_img_IDs)-1:
        val_batch_size += 1
        current_index += 1
    img_batch = np.zeros(shape=(val_batch_size, *PATCH_SIZE, 3))
    label_batch = np.zeros(shape=(val_batch_size, 3))
    while local_index < val_batch_size and continue_val:
        img_ID = val_img_IDs[current_val_index]
        img_batch[local_index] = img_to_array(load_img(img_ID))/255.
        label_batch[local_index] = val_labels[val_img_IDs[current_val_index]]
        local_index += 1
        current_val_index += 1
    if current_val_index+val_batch_size >= len(val_img_IDs):
        continue_val = False
    return img_batch, label_batch, continue_val


# custom angular error metric
def angular_error_metric(y_true, y_pred):
    return 180*tf.acos(K.clip(K.sum(K.l2_normalize(y_true, axis=-1) * K.l2_normalize(y_pred, axis=-1), axis=-1),
                              EPSILON, 1.-EPSILON))/np.pi


if __name__ == '__main__':
    imdb_dir = r'train\RECommended\imdb'
    model_dir = r'train\RECommended\models'
    logs_dir = os.path.join(model_dir, NETWORK)
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    print('{network:s} architecture is selected with batch size {batch_size:02d}, trained on {dataset:s} dataset.'.
          format(**{'network': NETWORK,
                    'dataset': DATASET,
                    'batch_size': BATCH_SIZE}))

    for current_fold in FOLDS:
        print('=' * 40)
        print('Cross validation: fold {} started.'.format(current_fold), flush=True)
        print('=' * 40)
        logs_dir_current_fold = os.path.join(logs_dir, 'fold_{}'.format(current_fold))
        if not os.path.exists(logs_dir_current_fold):
            os.makedirs(logs_dir_current_fold)

        # training data preparation
        train_img_IDs = []
        train_labels = dict()
        train_file = os.path.join(imdb_dir, 'fold_{}_train.txt'.format(current_fold))
        with open(train_file) as f:
            for line in f:
                img_ID = line.split('\t')[0]
                train_img_IDs.append(img_ID)
                gains = line.split('\t')[3]  # string type
                train_labels[img_ID] = [float(x) for x in gains.split(' ')]  # convert to float type

        # validation data preparation
        val_img_IDs, val_source_img_IDs = [], []
        val_labels = dict()
        val_file = os.path.join(imdb_dir, 'fold_{}_val.txt'.format(current_fold))
        with open(val_file) as f:
            for line in f:
                bias_angle = float(line.split('\t')[1])
                # for validation, only UNBIASED sub-images will be evaluate
                if bias_angle == 0:
                    img_ID = line.split('\t')[0]
                    source_img_ID = line.split('\t')[4]
                    val_img_IDs.append(img_ID)
                    val_source_img_IDs.append(source_img_ID)
                    gains = line.split('\t')[3]  # string type
                    val_labels[img_ID] = [float(x) for x in gains.split(' ')]  # convert to float type

        print('Data is ready. {} sub-images for training, {} for validation.'.format(len(train_img_IDs), len(val_img_IDs)))

        if NETWORK == 'Hierarchy-1':
            conv_layers_names = ['conv2d_1']
        elif NETWORK == 'Hierarchy-2':
            conv_layers_names = ['conv2d_1', 'conv2d_2']
        elif NETWORK == 'Hierarchy-3':
            conv_layers_names = ['conv2d_1', 'conv2d_2', 'conv2d_3']

        # load pre-trained weights in Inception-V3
        inception_model = applications.InceptionV3()
        # a dictionary records the layer name and layer weights in Inception-V3
        inception_layers = {layer.name: layer for layer in inception_model.layers}
        inception_weights = dict()
        for layer_name in conv_layers_names:
            inception_weights[layer_name] = inception_layers[layer_name].get_weights()
        K.clear_session()

        # create a model and initialize with inception_weights
        model = model_builder(level=args.level,
                              input_shape=(*PATCH_SIZE, 3))
        model_layers = {layer.name: layer for layer in model.layers}
        for layer_name in conv_layers_names:
            idx = list(model_layers.keys()).index(layer_name)
            model.layers[idx].set_weights(inception_weights[layer_name])
            print('Initialize {0} layer with weights in Inception v3.'.format(layer_name))

        model.compile(loss='mse',
                      optimizer=Nadam(lr=LR),
                      metrics=[angular_error_metric])
        model.summary()

        # uncomment following lines to plot the model architecture
        # from keras.utils import plot_model
        # plot_model(model, to_file=os.path.join(logs_dir, 'architecture.pdf'), show_shapes=True)

        # figure preparation
        fig = plt.figure()
        ax_mse = fig.add_subplot(111)
        ax_ang = ax_mse.twinx()
        eps = []

        history_train_mse, history_train_angular_errors = [], []
        history_val_mean_angular_errors, history_val_median_angular_errors = [], []
        min_median_angular_error = float('inf')
        for current_epoch in range(1, EPOCHS + 1):
            start_time = timer()
            print('=' * 60)
            print('Epoch {}/{} started.'.format(current_epoch, EPOCHS))

            # learning rate decrease
            if len(history_val_median_angular_errors) > PATIENCE:
                if np.min(history_val_median_angular_errors[-PATIENCE:]) > history_val_median_angular_errors[-PATIENCE-1] - MIN_DELTA:
                    old_lr = float(K.get_value(model.optimizer.lr))
                    new_lr = max(old_lr * FACTOR, MIN_LR)
                    K.set_value(model.optimizer.lr, new_lr)

            # learning rate increase
            if len(history_val_median_angular_errors) > PATIENCE * 10:
                if np.min(history_val_median_angular_errors[-PATIENCE*10:]) > history_val_median_angular_errors[-PATIENCE*10-1] - MIN_DELTA:
                    old_lr = float(K.get_value(model.optimizer.lr))
                    new_lr = min(old_lr * 2, MAX_LR)
                    K.set_value(model.optimizer.lr, new_lr)
                    print('Learning rate increased!')

            print('Learning rate in current epoch: {0:.2e}'.format(float(K.get_value(model.optimizer.lr))))

            train_mse, train_angular_errors = [], []
            val_angular_errors = []

            current_train_index = 0
            current_val_index = 0
            continue_train = True
            continue_val = True

            indices = np.arange(len(train_img_IDs))
            np.random.shuffle(indices)
            train_img_IDs = [train_img_IDs[i] for i in indices]

            # training phase
            while continue_train:
                b, l, continue_train = get_train_batch()
                logs = model.train_on_batch(b, l)
                train_mse.append(logs[0])
                train_angular_errors.append(logs[1])

            # validation phase
            while continue_val:
                b, l, continue_val = get_val_batch()
                if b.shape[0] > 4:  # only test on images with more than 4 crops 
                    estimates = model.predict_on_batch(b)
                    estimates /= estimates[:, 1][:, np.newaxis]
                    estimates = np.median(estimates, axis=0)
                    val_angular_errors.append(angular_error(l[0, ], estimates))
                else:
                    pass

            mean_val_angular_error_current_epoch = np.mean(val_angular_errors)
            median_val_angular_error_current_epoch = np.median(val_angular_errors)
            tri_val_angular_error_current_epoch = (np.percentile(val_angular_errors, 25) +
                                                   2 * np.median(val_angular_errors) +
                                                   np.percentile(val_angular_errors, 75)) / 4.
            b25_val_angular_error_current_epoch = percentile_mean(np.array(val_angular_errors), 0, 25)
            w25_val_angular_error_current_epoch = percentile_mean(np.array(val_angular_errors), 75, 100)

            print('MSE on training set: {0:.5f}(mean), {1:.5f}(median)'.format(np.mean(train_mse),
                                                                               np.median(train_mse)))
            print('Angular error on training set: {0:.3f}(mean), {1:.3f}(median)'.format(np.mean(train_angular_errors),
                                                                                         np.median(train_angular_errors)))
            print('Monitored angular error on validation set: {0:.3f}(mean), {1:.3f}(median), {2:.3f}(tri), {3:.3f}(best 25), {4:.3f}(worst 25)'
                  .format(mean_val_angular_error_current_epoch,
                          median_val_angular_error_current_epoch,
                          tri_val_angular_error_current_epoch,
                          b25_val_angular_error_current_epoch,
                          w25_val_angular_error_current_epoch))

            # historical records
            history_train_mse.append(np.mean(train_mse))
            history_train_angular_errors.append(np.mean(train_angular_errors))
            history_val_mean_angular_errors.append(mean_val_angular_error_current_epoch)
            history_val_median_angular_errors.append(median_val_angular_error_current_epoch)

            # plot the loss
            eps.append(current_epoch)
            mse_train_line, = ax_mse.plot(eps, history_train_mse, 'r--')
            ax_mse.set_xlabel('Epoch')
            ax_mse.set_ylabel('MSE loss', color='r')
            ax_mse.tick_params('y', colors='r')
            train_angular_error_line, = ax_ang.plot(eps, history_train_angular_errors, 'b--')
            val_mean_angular_error_line, = ax_ang.plot(eps, history_val_mean_angular_errors, 'b-')
            val_median_angular_error_line, = ax_ang.plot(eps, history_val_median_angular_errors, 'b:')
            ax_ang.set_ylabel('Angular loss', color='b')
            ax_ang.tick_params('y', colors='b')
            plt.legend((mse_train_line,
                        train_angular_error_line, val_mean_angular_error_line, val_median_angular_error_line),
                       ('MSE training loss', 'Angular training loss',
                        'Angular validation loss (mean)', 'Angular validation loss (median)'))
            plt.savefig(os.path.join(logs_dir_current_fold, "train_history.pdf"))

            # save model
            if median_val_angular_error_current_epoch < min_median_angular_error:
                min_median_angular_error = median_val_angular_error_current_epoch
                model.save_weights(os.path.join(logs_dir_current_fold,
                                                'epoch{epoch:02d}_'
                                                '{mean_angular_error:.3f}(mean)_'
                                                '{median_angular_error:.3f}(median)_'
                                                '{tri_angular_error:.3f}(tri)_'
                                                '{b25_angular_error:.3f}(b25)_'
                                                '{w25_angular_error:.3f}(w25).h5').format(
                    **{'epoch': current_epoch,
                       'mean_angular_error': mean_val_angular_error_current_epoch,
                       'median_angular_error': median_val_angular_error_current_epoch,
                       'tri_angular_error': tri_val_angular_error_current_epoch,
                       'b25_angular_error': b25_val_angular_error_current_epoch,
                       'w25_angular_error': w25_val_angular_error_current_epoch}))

            end_time = timer()
            print('{0:.0f}s elapsed'.format(end_time-start_time))

            # early-stopping
            if len(history_val_median_angular_errors) > EARLY_STOP_PATIENCE:
                if np.min(history_val_median_angular_errors[-EARLY_STOP_PATIENCE:]) > history_val_median_angular_errors[-EARLY_STOP_PATIENCE - 1] - MIN_DELTA:
                    print('No improvement detected. Stop the training.')
                    print('=' * 60)
                    break

        K.clear_session()

