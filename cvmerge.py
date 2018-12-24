# -*- coding: utf-8 -*-
import os
import re
import argparse

parser = argparse.ArgumentParser(description="Merge cross validation results.",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("cv_dir", type=str,
                    help="The directory containing models for cross validation. "
                         "e.g., train\\RECommended\\models\\Hierarchy-1")
args = parser.parse_args()

import numpy as np
import glob

from model import model_builder
from config import get_dataset_config
from utils import (get_ground_truth_dict,
                   get_masks_dict,
                   img2batch,
                   angular_error,
                   local_estimates_aggregation_naive,
                   local_estimates_aggregation,
                   percentile_mean)

# load configuration based on the pre-trained dataset
dataset_config = get_dataset_config(dataset='R')
# configurations
##############################
DATASET = dataset_config['dataset']
PATCHES = dataset_config['patches']  # the number of square sub-images
PATCH_SIZE = dataset_config['patch_size']  # the size of square sub-image
CONFIDENCE_THRESHOLD = dataset_config['confidence_threshold']
MODEL_DIR = dataset_config['model_dir']  # pre-trained model directory
INPUT_BITS = dataset_config['input_bits']  # bit length of the input image
VALID_BITS = dataset_config['valid_bits']  # valid bit length of the data
DARKNESS = dataset_config['darkness']  # black level
COLOR_CORRECTION_MATRIX = dataset_config['color_correction_matrix']
GAMMA = None
BATCH_SIZE = 64
NB_FOLDS = 3
##############################


def search_best_epoch(models_dir):
    min_median_angular_error = float('inf')
    best_model_dir = None
    trained_models = glob.glob(models_dir + '/*.h5')
    for model_name in trained_models:
        current_median_angular_error = float(re.search('\(mean\)_(.*)\(median\)', model_name).group(1))
        if current_median_angular_error <= min_median_angular_error:
            min_median_angular_error = current_median_angular_error
            best_model_dir = model_name
    return best_model_dir


def inference(model_level, model_dir, test_img_IDs):
    confidence_estimation_mode = False
    model = model_builder(level=model_level,
                          confidence=False,
                          input_shape=(*PATCH_SIZE, 3))
    model.load_weights(model_dir)
    ground_truth_dict = get_ground_truth_dict(r'train\RECommended\ground-truth.txt')
    masks_dict = get_masks_dict(r'train\RECommended\masks.txt')
    angular_errors_statistics = []
    for (counter, test_img_ID) in enumerate(test_img_IDs):
        print('Processing {}/{} images...'.format(counter + 1, len(test_img_IDs)), end='\r')
        # data generator
        batch, boxes, remained_boxes_indices, ground_truth = img2batch(test_img_ID,
                                                                       patch_size=PATCH_SIZE,
                                                                       input_bits=INPUT_BITS,
                                                                       valid_bits=VALID_BITS,
                                                                       darkness=DARKNESS,
                                                                       ground_truth_dict=ground_truth_dict,
                                                                       masks_dict=masks_dict,
                                                                       gamma=GAMMA)
        nb_batch = int(np.ceil(PATCHES / BATCH_SIZE))
        batch_size = int(PATCHES / nb_batch)  # actual batch size
        local_estimates, confidences = np.empty(shape=(0, 3)), np.empty(shape=(0,))

        # use batch(es) to feed into the network
        for b in range(nb_batch):
            batch_start_index, batch_end_index = b * batch_size, (b + 1) * batch_size
            batch_tmp = batch[batch_start_index:batch_end_index, ]
            if confidence_estimation_mode:
                # the model requires 2 inputs when confidence estimation mode is activated
                batch_tmp = [batch_tmp, np.zeros((batch_size, 3))]
            outputs = model.predict(batch_tmp)  # model inference
            if confidence_estimation_mode:
                # the model produces 6 outputs when confidence estimation mode is on. See model.py for more details
                # local_estimates is the gain instead of illuminant color!
                local_estimates = np.vstack((local_estimates, outputs[4]))
                confidences = np.hstack((confidences, outputs[5].squeeze()))
            else:
                # local_estimates is the gain instead of illuminant color!
                local_estimates = np.vstack((local_estimates, outputs))
                confidences = None

        if confidence_estimation_mode:
            global_estimate = local_estimates_aggregation(local_estimates, confidences)
        else:
            global_estimate = local_estimates_aggregation_naive(local_estimates)

        global_rgb_estimate = 1. / global_estimate  # convert gain into rgb triplet

        global_angular_error = angular_error(ground_truth, global_rgb_estimate)
        angular_errors_statistics.append(global_angular_error)

    return np.array(angular_errors_statistics)


def cross_validation_collect(cross_validation_dir):
    assert 'Hierarchy-' in cross_validation_dir
    fold_dirs = glob.glob(cross_validation_dir + '/*')
    assert set([os.path.join(cross_validation_dir, 'fold_{}'.format(i)) for i in range(1, NB_FOLDS + 1)]) <= set(fold_dirs)
    hierarchical_level = int(re.search('Hierarchy-(\d)', cross_validation_dir).group(1))
    cv_statistics = np.empty((NB_FOLDS, 5))  # 5 metrics: mean, med, tri, b25, w25

    for current_fold in range(1, NB_FOLDS+1):
        print('Fold {}/{} started.'.format(current_fold, NB_FOLDS))
        current_fold_dir = os.path.join(cross_validation_dir, 'fold_{}'.format(current_fold))
        best_model_dir = search_best_epoch(current_fold_dir)
        test_img_list_path = r'train\RECommended\imdb\fold_{}_val.txt'.format(current_fold)

        test_img_IDs = []
        with open(test_img_list_path) as f:
            for line in f:
                test_img_IDs.append(line.split('\t')[-1].rstrip())
        test_img_IDs = list(set(test_img_IDs))

        angular_errors = inference(model_level=hierarchical_level,
                                   model_dir=best_model_dir,
                                   test_img_IDs=test_img_IDs)
        current_fold_statistics = [np.mean(angular_errors),
                                   np.median(angular_errors),
                                   (np.percentile(angular_errors, 25) + 2 * np.median(angular_errors) + np.percentile(angular_errors, 75)) / 4.,
                                   percentile_mean(angular_errors, 0, 25),
                                   percentile_mean(angular_errors, 75, 100)]

        cv_statistics[current_fold - 1, :] = np.array(current_fold_statistics)
        print('Validation results for fold {0}: '
              '{1:.3f}(mean), {2:.3f}(median), {3:.3f}(tri), {4:.3f}(best 25), {5:.3f}(worst 25)'.
              format(current_fold, *current_fold_statistics))
        print('=' * 60)
        with open(os.path.join(cross_validation_dir, 'cv_results.txt'), "a") as f:
            f.write('Validation results for fold {0}: '
                    '{1:.3f}(mean), {2:.3f}(median), {3:.3f}(tri), {4:.3f}(best 25), {5:.3f}(worst 25)\n'.
                    format(current_fold, *current_fold_statistics))

    cv_results = np.mean(cv_statistics, axis=0)
    with open(os.path.join(cross_validation_dir, 'cv_results.txt'), "a") as f:
        f.write('=' * 40 + '\n')
        f.write('Cross validation result: '
                '{0:.2f}(mean), {1:.2f}(median), {2:.2f}(tri), {3:.2f}(best 25), {4:.2f}(worst 25)\n'.format(*cv_results))
    return cv_results


if __name__ == '__main__':
    network = os.path.split(args.cv_dir)[1]
    print('Merge cross validation statistics for {} model'.format(network))
    print('=' * 60)
    cv_results = cross_validation_collect(args.cv_dir)
    print('\nCross validation result for {0} model: '
          '{1:.2f}(mean), {2:.2f}(median), {3:.2f}(tri), {4:.2f}(best 25), {5:.2f}(worst 25)'.format(network,
                                                                                                     *cv_results))


