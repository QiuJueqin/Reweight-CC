# -*- coding: utf-8 -*-
import os
import argparse

parser = argparse.ArgumentParser(description="Read image(s) and perform computational color constancy. "
                                             "See README and paper Color Constancy by Image Feature Maps Reweighting for more details.",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("img_path", type=str,
                    help="dateset direcroty.\n"
                         "Use wildcard '*' to load all images in the directory (multi-image mode).\n"
                         "e.g., c:\\foo.jpg or sample_images\\MultiCam\\*")
parser.add_argument("-d", "--dataset", type=str, choices=['MultiCam', 'RECommended'], default='MultiCam',
                    help="select pre-trained model for MultiCam dataset or ColorChecker RECommended dataset. (default: MultiCam)\n"
                         "Images in MultiCam dataset are device-independent,\n"
                         "so models pre-trained on this dataset are also suitable for images from other sources.\n")
parser.add_argument("-l", "--level", type=int, choices=[1, 3], default=3,
                    help="the number of hierarchical levels. (default: 3)\n")
parser.add_argument("-c", "--confidence",
                    help="use network with the confidence estimation branch and aggregate local estimates based on their confidence scores.",
                    action="store_true")
parser.add_argument("-g", "--gamma",
                    help="apply the inverse gamma correction to the (non-linear) input image(s).\n"
                         "Turn on this option only if the input image(s) has gone through post-processing (e.g., downloaded from Internet).",
                    action="store_true")
parser.add_argument("-s", "--save", type=int, choices=[0, 1, 2, 3, 4], default=0,
                    help="save option. (default: 1)\n"
                    	 "-s 0/--save 0: save nothing (only for inference time test).\n"
                         "-s 1/--save 1: save the corrected image(s) only.\n"
                         "-s 2/--save 2: save the corrected image(s) as well as the result(s) of the local estimates.\n"
                         "-s 3/--save 3: save the corrected image(s) as well as the intermediate feature maps (may be slow).\n"
                         "-s 4/--save 4: save all described above.")
parser.add_argument("-r", "--record",
                    help="write illuminant estimation results into a text file.",
                    action="store_true")
parser.add_argument("-b", "--batch", type=int, metavar='N', default=64,
                    help="-b N/--batch N: batch size (default: 64).\n"
                         "Decrease it if encounter memory allocations issue.")
args = parser.parse_args()

from timeit import default_timer as timer
import glob
import matplotlib.pyplot as plt

from config import *
from utils import (read_image,
                   img2batch,
                   angular_error,
                   get_ground_truth_dict,
                   get_masks_dict,
                   local_estimates_aggregation_naive,
                   local_estimates_aggregation,
                   write_records,
                   write_statistics)
from visualization import *
from model import model_builder

# load configuration based on the pre-trained dataset
dataset_config = get_dataset_config(dataset=args.dataset)
# network architecture selection
model_config = get_model_config(level=args.level, confidence=args.confidence)

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
BRIGHTNESS_SCALE = dataset_config['brightness_scale']  # scale image brightness for better visualization
COLOR_CORRECTION_MATRIX = dataset_config['color_correction_matrix']
NETWORK = model_config['network']
PRETRAINED_MODEL = model_config['pretrained_model']
# feature maps names for visualization
INPUT_FEATURE_MAPS_NAMES = model_config['input_feature_maps_names']
REWEIGHT_MAPS_NAMES = model_config['reweight_maps_names']
OUTPUT_FEATURE_MAPS_NAMES = model_config['output_feature_maps_names']
##############################

if os.path.isdir(args.img_path):
    args.img_path = os.path.join(args.img_path, '*')
img_dir = os.path.dirname(args.img_path)
# get all image paths
imgs_path = glob.glob(args.img_path)
imgs_path = [i for i in imgs_path if os.path.splitext(i)[1] in ('.png', '.jpg')]
if len(imgs_path) > 1:
    multiple_images_mode = True
else:
    multiple_images_mode = False

# inverse gamma correction
if args.gamma:
    inverse_gamma_correction_mode = True
    GAMMA = 2.2
else:
    inverse_gamma_correction_mode = False
    GAMMA = None

if args.dataset == 'R':
    inverse_gamma_correction_mode = False
    GAMMA = None

# confidence estimation branch
if args.confidence:
    confidence_estimation_mode = True
else:
    confidence_estimation_mode = False

# record mode
if args.record:
    record_mode = True
    record_file_path = os.path.join(img_dir, 'Records_' + NETWORK + '.txt')
else:
    record_mode = False

# search ground-truth.txt file which contains the ground-truth illuminant colors of images
ground_truth_path = os.path.join(img_dir, 'ground-truth.txt')
if os.path.exists(ground_truth_path):
    ground_truth_mode = True
    # ground_truth_dict is a dictionary with image IDs as keys and ground truth colors as values
    ground_truth_dict = get_ground_truth_dict(ground_truth_path)
else:
    ground_truth_mode = False
    ground_truth_dict = None

# search masks.txt file which contains the coordinates to be excluded
masks_path = os.path.join(img_dir, 'masks.txt')
if os.path.exists(masks_path):
    # masks_dict is a dictionary with image IDs as keys and coordiantes as values
    masks_dict = get_masks_dict(masks_path)
else:
    masks_dict = None

# import model and load pre-trained parameters
model = model_builder(level=args.level,
                      confidence=args.confidence,
                      input_shape=(*PATCH_SIZE, 3))
network_path = os.path.join(MODEL_DIR, PRETRAINED_MODEL)
model.load_weights(network_path)

print('=' * 110)
print('{network:s} architecture is selected with batch size {batch_size:02d} (pre-trained on {dataset:s} dataset).'.
      format(**{'network': NETWORK,
                'dataset': DATASET,
                'batch_size': args.batch}))

if ground_truth_dict is not None:
    print('Ground-truth file found.')
if masks_dict is not None:
    print('Masks file found.')
if args.save == 3 or args.save == 4:
    print('Generating intermediate feature maps may take long time (>5s/image). Keep your patience.')

# from keras.utils import plot_model #  uncomment these 2 lines to plot the model architecture, if needed
# plot_model(model, to_file=os.path.join(model_dir, network+'_architecture.pdf'), show_shapes=True)
# model.summary()  # uncomment this line to print model details, if needed

if __name__ == '__main__':
    print('Processing started...')
    angular_errors_statistics = []  # record angular errors for all test images
    inference_times = []
    for (counter, img_path) in enumerate(imgs_path):
        img_name = os.path.splitext(os.path.basename(img_path))[0]  # image name without extension
        print(img_name + ':', end=' ', flush=True)
        # data generator
        batch, boxes, remained_boxes_indices, ground_truth = img2batch(img_path,
                                                                       patch_size=PATCH_SIZE,
                                                                       input_bits=INPUT_BITS,
                                                                       valid_bits=VALID_BITS,
                                                                       darkness=DARKNESS,
                                                                       ground_truth_dict=ground_truth_dict,
                                                                       masks_dict=masks_dict,
                                                                       gamma=GAMMA)
        nb_batch = int(np.ceil(PATCHES / args.batch))
        batch_size = int(PATCHES / nb_batch)  # actual batch size
        local_estimates, confidences = np.empty(shape=(0, 3)), np.empty(shape=(0,))

        start_time = timer()
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
        
        end_time = timer()
        inference_times.append(end_time - start_time)

        local_rgb_estimates = 1. / local_estimates  # convert gains into rgb triplet
        local_rgb_estimates /= local_rgb_estimates.sum(axis=1, keepdims=True)
        global_rgb_estimate = 1. / global_estimate  # convert gain into rgb triplet
        global_rgb_estimate /= global_rgb_estimate.sum()

        if ground_truth_mode:
            local_angular_errors = angular_error(ground_truth, local_rgb_estimates)
            global_angular_error = angular_error(ground_truth, global_rgb_estimate)
            angular_errors_statistics.append(global_angular_error)
        else:
            local_angular_errors = global_angular_error = None

        # Save the white balanced image
        if args.save in [1, 2, 3, 4]:
            img = read_image(img_path=img_path,
                             input_bits=INPUT_BITS,
                             valid_bits=VALID_BITS,
                             darkness=DARKNESS,
                             gamma=GAMMA)
            wb_imgs_path = os.path.join(img_dir, 'white_balanced_images')
            if not os.path.exists(wb_imgs_path):
                os.mkdir(wb_imgs_path)
            wb_img = white_balance(input_img=np.clip(BRIGHTNESS_SCALE*img, 0., 1.),
                                   global_estimate=global_estimate,
                                   color_correction_matrix=COLOR_CORRECTION_MATRIX)
            wb_img_path = os.path.join(wb_imgs_path, img_name + '_wb.png')
            plt.imsave(wb_img_path, wb_img)

        # Save the result of the local estimates
        if args.save in [2, 4]:
            local_wb_imgs_path = os.path.join(img_dir, 'local_estimates_images')
            if not os.path.exists(local_wb_imgs_path):
                os.mkdir(local_wb_imgs_path)
            local_wb_img = generate_local_wb_visualization(input_img=np.clip(BRIGHTNESS_SCALE*img, 0., 1.),
                                                           local_estimates=local_estimates,
                                                           global_estimate=global_estimate,
                                                           boxes=boxes,
                                                           remained_boxes_indices=remained_boxes_indices,
                                                           confidences=confidences,
                                                           ground_truth=ground_truth,
                                                           local_angular_errors=local_angular_errors,
                                                           global_angular_error=global_angular_error,
                                                           color_correction_matrix=COLOR_CORRECTION_MATRIX)
            local_wb_img_path = os.path.join(local_wb_imgs_path, img_name + '_local_estimates.jpg')
            local_wb_img.save(local_wb_img_path)

        # Save the mosaic image of intermediate feature maps
        if args.save in [3, 4]:
            mosaic_img_dir = os.path.join(img_dir, 'intermediate_feature_maps')
            if not os.path.exists(mosaic_img_dir):
                os.mkdir(mosaic_img_dir)
            mosaic_img = generate_feature_maps_visualization(model=model,
                                                             input_img=img,
                                                             input_feature_maps_names=INPUT_FEATURE_MAPS_NAMES,
                                                             reweight_maps_names=REWEIGHT_MAPS_NAMES,
                                                             output_feature_maps_names=OUTPUT_FEATURE_MAPS_NAMES)
            mosaic_img_path = os.path.join(mosaic_img_dir, img_name + '_intermediate_maps.jpg')
            mosaic_img.save(mosaic_img_path)

        # Record illuminant estimation results into a text file
        if args.record:
            write_records(record_file_path=record_file_path,
                          img_path=img_path,
                          global_estimate=global_estimate,
                          ground_truth=ground_truth,
                          global_angular_error=global_angular_error)

        print('done. ({0:d}/{1:d})'.format(counter + 1, len(imgs_path)))

    # Record overall statistics into a text file
    if args.record and ground_truth_mode:
        write_statistics(record_file_path, angular_errors_statistics)
    if len(inference_times) > 1:
        print('Average inference time: {0:.0f}ms/image.'.format(1000 * np.mean(inference_times[1:])))
