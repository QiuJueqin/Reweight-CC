import numpy as np
from PIL import Image, ImageFont, ImageDraw
import keras.backend as K

from utils import color_correction

BACKGROUND_COLOR = (10, 10, 10, 160)
TEXT_COLOR = BOX_COLOR = (230, 230, 230)
FONT = ImageFont.truetype(font='arial', size=24)
EPSILON = 1E-9


def white_balance(input_img,
                  global_estimate,
                  color_correction_matrix=None):
    img_wb = input_img.copy()
    # normalize the gain, otherwise white-balanced image may suffer from overflowing
    global_gain = global_estimate / global_estimate.min()
    img_wb *= global_gain.reshape((1, 1, 3))
    img_wb = img_wb.clip(0., 1.)
    if color_correction_matrix is not None:
        img_wb = color_correction(img_wb, color_correction_matrix)
    img_wb **= 1./2.2

    return img_wb


def generate_local_wb_visualization(input_img,
                                    local_estimates,
                                    global_estimate,
                                    boxes,
                                    remained_boxes_indices=None,
                                    confidences=None,
                                    ground_truth=None,
                                    local_angular_errors=None,
                                    global_angular_error=None,
                                    color_correction_matrix=None):
    img_height, img_width, _ = input_img.shape
    if remained_boxes_indices is not None:
        valid_boxes_indices = np.where(remained_boxes_indices < 12)[0]
        local_estimates = local_estimates[valid_boxes_indices, :]
        boxes = boxes[valid_boxes_indices, :]
        if confidences is not None:
            confidences = confidences[valid_boxes_indices]
        if local_angular_errors is not None:
            local_angular_errors = local_angular_errors[valid_boxes_indices]

    local_rgb_estimates = 1. / local_estimates
    local_rgb_estimates /= local_rgb_estimates.sum(axis=1, keepdims=True)
    global_rgb_estimate = 1. / global_estimate
    global_rgb_estimate /= global_rgb_estimate.sum()

    nb_valid_patch = local_estimates.shape[0]
    local_wb_img = input_img.copy()
    for i in range(nb_valid_patch):
        # normalize the gains, otherwise white-balanced image may suffer from overflowing
        local_gains = local_estimates[i, :] / local_estimates[i, :].min()
        local_wb_img[boxes[i, 1]:boxes[i, 3], boxes[i, 0]:boxes[i, 2], :] *= local_gains.reshape((1, 1, 3))
    local_wb_img = local_wb_img.clip(0., 1.)
    if color_correction_matrix is not None:
        local_wb_img = color_correction(local_wb_img, color_correction_matrix)
    local_wb_img **= 1./2.2
    # add some labels
    local_wb_img = Image.fromarray((local_wb_img * 255).astype('uint8'))
    draw = ImageDraw.Draw(local_wb_img, 'RGBA')
    
    for i in range(nb_valid_patch):
        draw.rectangle([(boxes[i, 0], boxes[i, 1]), (boxes[i, 2], boxes[i, 3])], outline=BOX_COLOR)
        text = '[{0:.3f}, {1:.3f}, {2:.3f}]'.format(*local_rgb_estimates[i, :])
        w0, h0 = draw.textsize(text, FONT)
        draw.rectangle((boxes[i, 0], boxes[i, 1], boxes[i, 0] + w0, boxes[i, 1] + h0), fill=BACKGROUND_COLOR)
        draw.text((boxes[i, 0], boxes[i, 1]), text, fill=(230, 230, 230), font=FONT)
        if confidences is not None:
            text = 'Confidence {0:.3f}'.format(confidences[i])
            w1, h1 = draw.textsize(text, FONT)
            draw.rectangle((boxes[i, 0], boxes[i, 1] + h0, boxes[i, 0] + w1, boxes[i, 1] + h0 + h1),
                           fill=BACKGROUND_COLOR)
            draw.text((boxes[i, 0], boxes[i, 1] + h0), text, fill=TEXT_COLOR, font=FONT)
        else:
            w1 = h1 = 0
        if local_angular_errors is not None:
            text = 'Error: {0:.2f}°'.format(local_angular_errors[i])
            w2, h2 = draw.textsize(text, FONT)
            draw.rectangle((boxes[i, 0], boxes[i, 1] + h0 + h1, boxes[i, 0] + w2, boxes[i, 1] + h0 + h1 + h2),
                           fill=BACKGROUND_COLOR)
            draw.text((boxes[i, 0], boxes[i, 1] + h0 + h1), text, fill=TEXT_COLOR, font=FONT)
    # write the global information on the right bottom corner
    text = 'Global estimate: [{0:.3f}, {1:.3f}, {2:.3f}]'.format(*global_rgb_estimate)
    w0, h0 = draw.textsize(text, FONT)
    draw.rectangle((img_width - w0, img_height - h0, img_width, img_height), fill=BACKGROUND_COLOR)
    draw.text((img_width - w0, img_height - h0), text, fill=(230, 230, 230), font=FONT)
    if (ground_truth is not None) and (global_angular_error is not None):
        text = 'Ground-truth: [{0:.3f}, {1:.3f}, {2:.3f}]'.format(*ground_truth)
        w1, h1 = draw.textsize(text, FONT)
        draw.rectangle((img_width - w1, img_height - h0 - h1, img_width, img_height - h0), fill=BACKGROUND_COLOR)
        draw.text((img_width - w1, img_height - h0 - h1), text, fill=TEXT_COLOR, font=FONT)
        text = 'Global error: {0:.2f}°'.format(global_angular_error)
        w2, h2 = draw.textsize(text, FONT)
        draw.rectangle((img_width - w2, img_height - h0 - h1 - h2, img_width, img_height - h0 - h1), fill=BACKGROUND_COLOR)
        draw.text((img_width - w2, img_height - h0 - h1 - h2), text, fill=TEXT_COLOR, font=FONT)

    return local_wb_img


def get_intermediate_output(model, input_img, layer_idx):
    """
    get the output of an intermediate layer in a keras model
    :param model: keras model
    :param input_img: input image tensor as Numpy array
    :param layer_idx: index of the target layer
    :return: output tensor of intermediate layer as Numpy array
    """
    if input_img.ndim == 3:
        input_img = np.expand_dims(input_img, axis=0)  # convert into batch format
    f = K.function([model.layers[0].input, K.learning_phase()],
                   [model.layers[layer_idx].output])
    # the second argument denotes the learning phase (0 for test mode and 1 for train mode)
    intermediate_output = f([input_img, 0])[0].squeeze()
    if intermediate_output.ndim == 2:
        intermediate_output = np.expand_dims(intermediate_output, axis=-1).repeat(3, axis=-1)
    elif intermediate_output.ndim == 3:
        # remain 3 channels with max activations, if the #channels > 3
        if intermediate_output.shape[-1] > 3:
            max_activation_channels_indices = np.sum(intermediate_output, axis=(0, 1)).argsort()[::-1][:3]
            intermediate_output = intermediate_output[:, :, max_activation_channels_indices]

    return intermediate_output


def generate_feature_maps_visualization(model,
                                        input_img,
                                        input_feature_maps_names,
                                        reweight_maps_names,
                                        output_feature_maps_names):
    """
    generate a mosaic image with intermediate feature maps
    :param model: keras model
    :param input_img: input image tensor as Numpy array
    :param input_feature_maps_names: list of the input feature maps of the ReWU
    :param reweight_maps_names: list of the reweight map
    :param output_feature_maps_names: list of the output feature maps of the ReWU
    :return: the mosaic image as Image object
    """
    model_layers_dict = {layer.name: layer for layer in model.layers}
    input_feature_maps, reweight_maps, output_feature_maps = dict(), dict(), dict()
    for layer_name in model_layers_dict:
        layer_idx = list(model_layers_dict.keys()).index(layer_name)
        if layer_name in input_feature_maps_names:
            input_feature_maps[layer_name] = get_intermediate_output(model, input_img, layer_idx)
        elif layer_name in reweight_maps_names:
            reweight_maps[layer_name] = get_intermediate_output(model, input_img, layer_idx)
        elif layer_name in output_feature_maps_names:
            output_feature_maps[layer_name] = get_intermediate_output(model, input_img, layer_idx)

    thumbnail_size = list(map(lambda x: x.shape, list(reweight_maps.values())))[-1]
    thumbnail_size = (thumbnail_size[1], thumbnail_size[0])  # in [width, height] format
    margin = 10
    mosaic_img = Image.new('RGB', (3 * thumbnail_size[0] + 4 * margin,
                                   len(output_feature_maps) * (thumbnail_size[1] + margin) + margin))
    input_img_thumbnail = Image.fromarray((input_img * 255).astype('uint8'))
    input_img_thumbnail.thumbnail(thumbnail_size)
    mosaic_img.paste(im=input_img_thumbnail, box=(margin, margin))
    h = thumbnail_size[1] + 2 * margin
    for layer_name in input_feature_maps:
        tmp_map = input_feature_maps[layer_name]
        tmp_map = tmp_map.clip(0., None) / tmp_map.clip(0., None).max()
        tmp_map = Image.fromarray((tmp_map * 255).astype('uint8'))
        tmp_map.thumbnail(thumbnail_size)
        mosaic_img.paste(im=tmp_map, box=(0, h))
        h += (thumbnail_size[1] + margin)
    h = margin
    for layer_name in reweight_maps:
        tmp_map = reweight_maps[layer_name]
        tmp_map = tmp_map.clip(0., None)
        tmp_map *= 1 / (np.percentile(tmp_map, 95) + EPSILON)
        tmp_map = tmp_map.clip(0., 1.) ** (1 / 2.2)  # gamma for better visualization
        tmp_map = Image.fromarray((tmp_map * 255).astype('uint8'))
        tmp_map.thumbnail(thumbnail_size)
        mosaic_img.paste(im=tmp_map, box=(thumbnail_size[0] + 2 * margin, h))
        h += (thumbnail_size[1] + margin)
    h = margin
    for layer_name in output_feature_maps:
        tmp_map = output_feature_maps[layer_name]
        tmp_map = tmp_map.clip(0., None)
        tmp_map *= 1 / (np.percentile(tmp_map, 95) + EPSILON)
        tmp_map = tmp_map.clip(0., 1.) ** (1 / 2.2)  # gamma for better visualization
        tmp_map = Image.fromarray((tmp_map * 255).astype('uint8'))
        tmp_map.thumbnail(thumbnail_size)
        mosaic_img.paste(im=tmp_map, box=(2 * thumbnail_size[0] + 3 * margin, h))
        h += (thumbnail_size[1] + margin)

    return mosaic_img
