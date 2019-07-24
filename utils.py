import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import path
from PIL import Image
from keras.utils import Sequence
from keras.preprocessing.image import img_to_array, load_img


class DataGenerator(Sequence):
    """Generates data for Keras"""
    def __init__(self, list_imgs, labels, batch_size=64, dim=(224, 224, 3), shuffle=True):
        self.list_imgs = list_imgs
        self.labels = labels
        self.batch_size = batch_size
        self.dim = dim
        self.shuffle = shuffle
        self.indexes = None
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_imgs) / self.batch_size))

    def __getitem__(self, index):
        """Generate one batch of data"""
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of images
        list_imgs_temp = [self.list_imgs[k] for k in indexes]
        # Generate data
        x, y = self.__data_generation(list_imgs_temp)
        return x, y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.list_imgs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_imgs_temp):
        """Generates data containing batch_size samples # X : (n_samples, *dim, n_channels)"""
        # Initialization
        x = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, 3), dtype='float32')
        # Generate data
        for i, img_path in enumerate(list_imgs_temp):
            # store image
            x[i, ] = img_to_array(load_img(img_path))/255.
            # store gains
            y[i, ] = self.labels[img_path]
        return x, y


def read_image(img_path,
               input_bits=8,
               valid_bits=8,
               darkness=0.,
               gamma=None):
    """
    read image from disk and return a Numpy array
    :param img_path: image path
    :param input_bits: the bit length of the input image, usually 8 or 16.
    :param valid_bits: the actual bit length of the image, e.g., 8, 10, 12, 14,...
    :param darkness: the black level of the input image.
    :param gamma: apply inverse gamma correction (gamma=gamma) to the input image.
    :return: image data as Numpy array
    """
    img = plt.imread(img_path)
    if img.dtype == 'uint8':
        img = (img * (2 ** input_bits - 1.) - darkness) / (2 ** valid_bits - 1.) / 255.
    else:
        img = (img * (2 ** input_bits - 1.) - darkness) / (2 ** valid_bits - 1.)
    img = img.clip(0., 1.)
    if gamma is not None:
        img **= gamma
    return img


def img2batch(img_path,
              patch_size=(224, 224),
              input_bits=8,
              valid_bits=8,
              darkness=0,
              ground_truth_dict=None,
              masks_dict=None,
              gamma=None):
    """
    Sample sub-images from the input full-resolution image and return a P*H*W*3 tensor,
    where P is the number of sub-images, H and W are height and width of each sub-image.
    In this script we fixed all these parameters: P = 18, H = W = 224.
    :param img_path: the path of the input full-resolution image.
    :param patch_size: the target size of the sub-image.
    :param input_bits: the bit length of the input image, usually 8 or 16.
    :param valid_bits: the actual bit length of the image, e.g., 8, 10, 12, 14,...
    :param darkness: the black level of the input image.
    :param ground_truth_dict: the dictionary containing the ground truth illuminant colors.
    :param masks_dict: the dictionary containing the coordinates that should be removed from the sub-images.
    :param gamma: apply inverse gamma correction (gamma=gamma) to the input image.
    :return:
    patches: P*H*W*3 tensor of the sub-images as Numpy array
    boxes: coordinates of sub-images,
    remained_boxes_indices: the indices of sub-images to be visualized
    ground_truth: ground truth color of the illuminant in the input image (if provided)
    Please see README.md for more detailed information.
    """
    n_x, n_y = 4, 3
    n_patches = n_x * n_y + (n_x-1) * (n_y-1)  # number of sub-images
    img = read_image(img_path,
                     input_bits=input_bits,
                     valid_bits=valid_bits,
                     darkness=darkness,
                     gamma=gamma)

    h, w, _ = img.shape
    if h > w:
        n_x, n_y = n_y, n_x  # for portrait mode
    patch_size_orig = int(min(w/n_x, h/n_y))  # size of sub-image before resizing
    assert patch_size_orig >= 224, "The image size is too small to produce patches with lengths greater than 224px."
    x1, y1 = np.meshgrid(range(int((w - patch_size_orig*n_x)/2), int(w - int((w - patch_size_orig*n_x)/2) - 1), patch_size_orig),
                         range(int((h - patch_size_orig*n_y)/2), int(h - int((h - patch_size_orig*n_y)/2) - 1), patch_size_orig))
    x2, y2 = np.meshgrid(range(int((w - patch_size_orig * n_x) / 2 + patch_size_orig/2),
                               int(w - int((w - patch_size_orig * n_x) / 2 + patch_size_orig/2) - patch_size_orig/2 - 1), patch_size_orig),
                         range(int((h - patch_size_orig * n_y) / 2 + patch_size_orig/2),
                               int(h - int((h - patch_size_orig * n_y) / 2 + patch_size_orig/2) - patch_size_orig/2 - 1), patch_size_orig))
    x = np.hstack([x1.flatten(), x2.flatten()])
    y = np.hstack([y1.flatten(), y2.flatten()])
    del x1, y1, x2, y2

    # remove sub-images that contains the points recorded in the masks_dict
    if (masks_dict is not None) and (os.path.basename(img_path) in masks_dict):
        mask_coordinates = masks_dict[os.path.basename(img_path)]
        remained_boxes_indices = []
        idx = 0
        for _x, _y in zip(x, y):
            vertices = path.Path([(_x, _y),
                                  (_x + patch_size_orig, _y),
                                  (_x + patch_size_orig, _y + patch_size_orig),
                                  (_x, _y + patch_size_orig)])
            isinsquare = vertices.contains_points(mask_coordinates)
            if np.all(np.logical_not(isinsquare)):
                remained_boxes_indices.append(idx)
            idx += 1
        remained_boxes_indices = np.array(remained_boxes_indices)
        n_patches = len(remained_boxes_indices)
        # if the number of remained sub-images is smaller than 3, ignore the masks
        if n_patches >= 3:
            x, y = x[remained_boxes_indices], y[remained_boxes_indices]
        else:
            n_patches = n_x * n_y + (n_x-1) * (n_y-1)
            remained_boxes_indices = np.array(range(n_patches))
    else:
        remained_boxes_indices = np.array(range(n_patches))

    patches = np.empty((n_patches, *patch_size, 3))  # tensor of sub-images
    boxes = np.empty((n_patches, 4), dtype='int16')  # coordinate of boxes, in [x0, y0, x0+width, y0+height] format
    p = 0
    for _x, _y in zip(x, y):
        patch = img[_y:_y+patch_size_orig, _x:_x+patch_size_orig, :]
        patch = Image.fromarray((patch * 255).astype('uint8'), mode='RGB')
        patch.thumbnail(patch_size, Image.BICUBIC)  # image resizing
        patches[p, ] = (np.array(patch).astype('float32') / 255.).clip(0., 1.)
        boxes[p, ] = np.array([_x, _y, _x+patch_size_orig, _y+patch_size_orig])
        p += 1

    if (ground_truth_dict is not None) and (os.path.basename(img_path) in ground_truth_dict):
        ground_truth = ground_truth_dict[os.path.basename(img_path)]
    else:
        ground_truth = None

    return patches, boxes, remained_boxes_indices, ground_truth


def angular_error(ground_truth, prediction):
    """
    calculate angular error(s) between the ground truth RGB triplet(s) and the predicted one(s)
    :param ground_truth: N*3 or 1*3 Numpy array, each row for one ground truth triplet
    :param prediction: N*3 Numpy array, each row for one predicted triplet
    :return: angular error(s) in degree as Numpy array
    """
    ground_truth_norm = ground_truth / np.linalg.norm(ground_truth, ord=2, axis=-1, keepdims=True)
    prediction_norm = prediction / np.linalg.norm(prediction, ord=2, axis=-1, keepdims=True)
    return 180 * np.arccos(np.sum(ground_truth_norm * prediction_norm, axis=-1).clip(0., 1.)) / np.pi


def get_ground_truth_dict(ground_truth_path):
    """
    read ground-truth illuminant colors from text file, in which each line is in the 'ID  r g b' format
    :param ground_truth_path: path of the ground-truth.txt file
    :return: a dictionary with image IDs as keys and ground truth rgb colors as values
    """
    ground_truth_dict = dict()
    with open(ground_truth_path) as f:
        for line in f:
            img_id = line.split('\t')[0]
            illuminant_rgb = line.split('\t')[1]  # string type
            ground_truth_dict[img_id] = np.array([float(char) for char in illuminant_rgb.split(' ')])
    return ground_truth_dict


def get_masks_dict(masks_path):
    """
    read coordinates from text file, in which each line is in the 'ID  x1 x2 x3...  y1 y2 y3...' format
    :param masks_path: path of the masks.txt file
    :return: a dictionary with image IDs as keys and coordinates as values
    """
    masks_dict = dict()
    with open(masks_path) as f:
        for line in f:
            img_id = line.split('\t')[0]
            coordinates_x = line.split('\t')[1]  # string type
            coordinates_y = line.split('\t')[2]  # string type
            coordinates_x = np.array([float(char) for char in coordinates_x.split(' ')]).reshape((-1, 1))
            coordinates_y = np.array([float(char) for char in coordinates_y.split(' ')]).reshape((-1, 1))
            masks_dict[img_id] = np.hstack([coordinates_x, coordinates_y])
    return masks_dict


def local_estimates_aggregation_naive(local_estimates):
    """
    aggregate local illuminant color estimates into a global estimate
    :param local_estimates: N*3 numpy array, each row is a local estimate in [R_gain, G_gain, B_gain] format
    :return: 1*3 numpy array, the aggregated global estimate
    """
    local_estimates_norm = local_estimates / np.expand_dims(local_estimates[:, 1], axis=-1)
    global_estimate = np.median(local_estimates_norm, axis=0)
    return global_estimate / global_estimate.sum()


def local_estimates_aggregation(local_estimates, confidences):
    """
    aggregate local illuminant color estimates into a global estimate
    Note: the aggregation method here is kind of different from that described in the paper
    :param local_estimates: N*3 numpy array, each row is a local estimate in [R_gain, G_gain, B_gain] format
    :param confidences: (N, ) numpy array recording the confidence scores of N local patches
    :return: 1*3 numpy array, the aggregated global estimate
    """
    reliable_patches_indices = np.where((confidences > np.median(confidences)) & (confidences > 0.5))[0]
    if confidences.max() > 0.5 and len(reliable_patches_indices) >= 2:
        local_estimates_confident = local_estimates[reliable_patches_indices, :]
        local_estimates_confident_norm = local_estimates_confident / np.expand_dims(local_estimates_confident[:, 1], axis=-1)
        global_estimate = np.median(local_estimates_confident_norm, axis=0)
        return global_estimate / global_estimate.sum()
    else:
        return local_estimates_aggregation_naive(local_estimates)


def color_correction(img, color_correction_matrix):
    """
    use color correction matrix to correct image
    :param img: image to be corrected
    :param color_correction_matrix: 3*3 color correction matrix which has been normalized
    such that the sum of each row is equal to 1
    :return:
    """
    color_correction_matrix = np.asarray(color_correction_matrix)
    h, w, _ = img.shape
    img = np.reshape(img, (h * w, 3))
    img = np.dot(img, color_correction_matrix.T)
    return np.reshape(img, (h, w, 3)).clip(0., 1.)


def percentile_mean(x, prctile_lower, prctile_upper):
    """
    calculate the mean of elements within a percentile range.
    can be used to calculate the 'best x%' or 'worst x%' accuracy of a color constancy algorithm
    :param x: input numpy array
    :param prctile_lower: the lower limit of the percentile range
    :param prctile_upper: the upper limit of the percentile range
    :return: the arithmetic mean of elements within the percentile range [prctile_lower, prctile_upper]
    """
    if len(x) == 1:
        return x[0]
    else:
        x_sorted = np.sort(x)
        element_start_index = int(len(x) * prctile_lower / 100)
        element_end_index = int(len(x) * prctile_upper / 100)
        return x_sorted[element_start_index:element_end_index].mean()


def write_records(record_file_path,
                  img_path,
                  global_estimate,
                  ground_truth=None,
                  global_angular_error=None):
    """
    write one illuminant estimation entry in to the text file
    :param record_file_path: text file path
    :param img_path: source image path
    :param global_estimate: the estimated illuminant color
    :param ground_truth: the ground truth illuminant color
    :param global_angular_error: the angular error between ground_truth and global_estimate
    :return: None
    """
    with open(record_file_path, "a") as f:
        f.write("{0}\t".format(os.path.basename(img_path)))
        global_rgb_estimate = 1. / global_estimate
        global_rgb_estimate /= global_rgb_estimate.sum()
        f.write("Global estimate: [{0:.4f}, {1:.4f}, {2:.4f}]\t".format(*global_rgb_estimate))
        if (ground_truth is not None) and (global_angular_error is not None):
            ground_truth_rgb = ground_truth / ground_truth.sum()
            f.write("Ground-truth: [{0:.4f}, {1:.4f}, {2:.4f}]\t".format(*ground_truth_rgb))
            f.write("Angular error: {0:.2f}\t".format(global_angular_error))
        f.write('\n')


def write_statistics(record_file_path, angular_error_statistics):
    """
    write illuminant estimation results in to the text file
    :param record_file_path: text file path
    :param angular_error_statistics: list of angular errors for all test images
    :return: None
    """
    angular_error_statistics = np.asarray(angular_error_statistics)
    angular_errors_mean = angular_error_statistics.mean()
    angular_errors_median = np.median(angular_error_statistics)
    angular_errors_trimean = (np.percentile(angular_error_statistics, 25) + 2 * np.median(angular_error_statistics) + np.percentile(angular_error_statistics, 75)) / 4.
    angular_errors_best_quarter = percentile_mean(angular_error_statistics, 0, 25)
    angular_errors_worst_quarter = percentile_mean(angular_error_statistics, 75, 100)
    with open(record_file_path, "a") as f:
        f.write("Angular error metrics (in degree):\n"
                "mean: {angular_errors_mean:.2f}, "
                "median: {angular_errors_median:.2f}, "
                "trimean: {angular_errors_trimean:.2f}, "
                "best 25%: {angular_errors_best_quarter:.2f}, "
                "worst 25%: {angular_errors_worst_quarter:.2f} "
                "({nb_imgs:d} images)".format(**{'angular_errors_mean': angular_errors_mean,
                                                 'angular_errors_median': angular_errors_median,
                                                 'angular_errors_trimean': angular_errors_trimean,
                                                 'angular_errors_best_quarter': angular_errors_best_quarter,
                                                 'angular_errors_worst_quarter': angular_errors_worst_quarter,
                                                 'nb_imgs': len(angular_error_statistics)}))


def convert_back(record_file_path):
    """
    convert estimated illuminant colors (and the ground truth, if necessary) back into individual camera color spaces
    such that the angular errors could be more comparable with those in other literatures.
    THIS FUNCTION WORKS ONLY FOR MULTICAM DATASET!
    :param record_file_path: text file path, same as that in write_statistics function
    :return: None
    """
    record_file_path_cam_colorspace = record_file_path.replace('.txt', '_camcolorspace.txt')
    errors_in_cam_colorspace = []

    with open(record_file_path, "r") as f:
        for line in f:
            [img_path, groundtruth, prediction] = get_line_info(line)
            if not line: break
            ccm = get_ccm(get_camera_model(img_path))
            groundtruth_in_cam_colorspace = groundtruth * ccm^(-1)
            prediction_in_cam_colorspace = prediction * ccm^(-1)
            errors_in_cam_colorspace.append(angular_error(groundtruth_in_cam_colorspace, prediction_in_cam_colorspace))
            write_records(record_file_path_cam_colorspace,
                          img_path,
                          prediction_in_cam_colorspace,
                          ground_truth=groundtruth_in_cam_colorspace,
                          global_angular_error=errors_in_cam_colorspace[-1])
    write_statistics(record_file_path_cam_colorspace, errors_in_cam_colorspace)


def get_line_info(line):
    s = line.split('\t')
    if len(s) != 4:
        return None
    img_path = s[0]
    prediction = [float(x) for x in re.split('(\d+\.?\d*)', s[1])[1:-1:2]]
    groundtruth = [float(x) for x in re.split('(\d+\.?\d*)', s[2])[1:-1:2]]
    return img_path, groundtruth, prediction


def get_camera_model(img_name):
    if '_' in img_name:
        camera_model = img_name.split('_')[0]
        camera_model = 'Canon5D' if camera_model == 'IMG'
    elif '8D5U' in img_name:
        camera_model = 'Canon1D'
    else:
        camera_model = 'Canon550D'


def get_ccm(camera_model):
    camera_models = ('Canon5D', 'Canon1D', 'Canon550D', 'Canon1DsMkIII',
                     'Canon600D', 'FujifilmXM1', 'NikonD5200', 'OlympusEPL6',
                     'PanasonicGX1', 'SamsungNX2000', 'SonyA57')
    if camera_model not in camera_models:
        return None
    # extracted from dcraw.c
    matrices = ((6347,-479,-972,-8297,15954,2480,-1968,2131,7649), # Canon 5D
                (4374,3631,-1743,-7520,15212,2472,-2892,3632,8161), # Canon 1Ds
                (6941,-1164,-857,-3825,11597,2534,-416,1540,6039),  # Canon 550D
                (5859,-211,-930,-8255,16017,2353,-1732,1887,7448),  # Canon 1Ds Mark III
                (6461,-907,-882,-4300,12184,2378,-819,1944,5931),   # Canon 600D
                (10413,-3996,-993,-3721,11640,2361,-733,1540,6011), # FujifilmXM1
                (8322,-3112,-1047,-6367,14342,2179,-988,1638,6394), # Nikon D5200
                (8380,-2630,-639,-2887,10725,2496,-627,1427,5438),  # Olympus E-PL6
                (6763,-1919,-863,-3868,11515,2684,-1216,2387,5879), # Panasonic GX1
                (7557,-2522,-739,-4679,12949,1894,-840,1777,5311),  # SamsungNX2000
                (5991,-1456,-455,-4764,12135,2980,-707,1425,6701))    # Sony SLT-A57
    xyz2cam = np.asarray(matrices[camera_models.index(camera_model)])/10000
    xyz2cam = xyz2cam.reshape(3, 3).T
    linsRGB2XYZ = np.array((0.4124564, 0.3575761, 0.1804375),
                           (0.2126729, 0.7151522, 0.0721750),
                           (0.0193339, 0.1191920, 0.9503041))
    return xyz2cam.dot(linsRGB2XYZ).inverse.T # camera2linsRGB matrix