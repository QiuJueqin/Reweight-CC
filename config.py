def get_dataset_config(dataset):
    db_config = {'patches': 18,
                 'patch_size': (224, 224),
                 'confidence_threshold': 0.5}
    if dataset == 'R':
        db_config['dataset'] = r'ColorChecker RECommended'
        db_config['model_dir'] = r'pretrained_models\RECommended'
        db_config['input_bits'] = 16
        db_config['valid_bits'] = 12
        db_config['darkness'] = 129.
        db_config['brightness_scale'] = 4.
        db_config['color_correction_matrix'] = [[1.7494, -0.8470, 0.0976],
                                                [-0.1565, 1.4380, -0.2815],
                                                [0.0786, -0.5070, 1.4284]]
    else:
        db_config['dataset'] = r'MultiCam'
        db_config['model_dir'] = r'pretrained_models\MultiCam'
        db_config['input_bits'] = 8
        db_config['valid_bits'] = 8
        db_config['darkness'] = 0.
        db_config['brightness_scale'] = 1.
        db_config['color_correction_matrix'] = None

    return db_config


def get_model_config(level, confidence):
    model_config = dict()
    if level == 1:
        model_config['network'] = r'Hierarchy-1'
        model_config['input_feature_maps_names'] = ['activation_1']
        model_config['reweight_maps_names'] = ['activation_2', 'activation_3']
        model_config['output_feature_maps_names'] = ['multiply_1', 'multiply_2']
        model_config['LR'] = 5E-5
    elif level == 2:
        model_config['network'] = r'Hierarchy-2'
        model_config['input_feature_maps_names'] = ['activation_1', 'activation_2']
        model_config['reweight_maps_names'] = ['activation_3', 'activation_4', 'activation_5']
        model_config['output_feature_maps_names'] = ['multiply_1', 'multiply_2', 'multiply_3']
        model_config['LR'] = 4E-5
    elif level == 3:
        model_config['network'] = r'Hierarchy-3'
        model_config['input_feature_maps_names'] = ['activation_1', 'activation_2',
                                                    'activation_3']
        model_config['reweight_maps_names'] = ['activation_4', 'activation_5',
                                               'activation_6', 'activation_7']
        model_config['output_feature_maps_names'] = ['multiply_1', 'multiply_2',
                                                     'multiply_3', 'multiply_4']
        model_config['LR'] = 4E-5
    elif level == 5:
        model_config['network'] = r'Hierarchy-5'
        model_config['input_feature_maps_names'] = ['activation_1', 'activation_2', 'max_pooling2d_1',
                                                    'activation_4', 'activation_5']
        model_config['reweight_maps_names'] = ['activation_6', 'activation_7', 'activation_8',
                                               'activation_9', 'activation_10', 'activation_11']
        model_config['output_feature_maps_names'] = ['multiply_1', 'multiply_2', 'multiply_3',
                                                     'multiply_4', 'multiply_5', 'multiply_6']
        model_config['LR'] = 3E-5
    if confidence:
        model_config['network'] += '-Confidence'

    model_config['pretrained_model'] = model_config['network'] + '.h5'

    return model_config
