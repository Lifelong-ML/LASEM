from tensorflow.nn import relu

def num_data_points(data_type, data_percent):
    # train, valid, and test
    if 'mnist5' in data_type:
        if data_percent == 1:
            return (100, 20, 1800)
        elif data_percent == 3:
            return (300, 60, 1800)
        elif data_percent == 5:
            return (500, 100, 1800)
        elif data_percent == 7:
            return (700, 140, 1800)
        elif data_percent == 9:
            return (900, 180, 1800)
        elif data_percent == 10:
            return (1000, 200, 1800)
        elif data_percent == 30:
            return (3000, 600, 1800)
        elif data_percent == 50:
            return (5000, 1000, 1800)
        elif data_percent == 70:
            return (7000, 1000, 1800)
        elif data_percent == 90:
            return (9000, 1000, 1800)
        else:
            return (None, None, None)
    elif 'mnist10' in data_type:
        if data_percent == 1:
            return (100, 10, 2000)
        elif data_percent == 3:
            return (300, 30, 2000)
        elif data_percent == 5:
            return (500, 50, 2000)
        elif data_percent == 7:
            return (700, 70, 2000)
        elif data_percent == 9:
            return (900, 90, 2000)
        elif data_percent == 10:
            return (1000, 100, 2000)
        elif data_percent == 30:
            return (3000, 300, 2000)
        elif data_percent == 50:
            return (5000, 500, 2000)
        elif data_percent == 70:
            return (7000, 700, 2000)
        elif data_percent == 90:
            return (9000, 900, 2000)
        else:
            return (None, None, None)
    elif ('cifar10' in data_type) and not ('cifar100' in data_type):
        if data_percent == 2:
            return (160, 40, 2000)
        elif data_percent == 4:
            return (340, 60, 2000)
        elif data_percent == 6:
            return (500, 100, 2000)
        elif data_percent == 8:
            return (660, 140, 2000)
        elif data_percent == 10:
            return (840, 160, 2000)
        elif data_percent == 30:
            return (2500, 500, 2000)
        elif data_percent == 50:
            return (4160, 840, 2000)
        elif data_percent == 70:
            return (5840, 1160, 2000)
        elif data_percent == 90:
            return (7500, 1500, 2000)
        elif data_percent == 100:
            return (8330, 1670, 2000)
        else:
            return (None, None, None)
    elif 'cifar100' in data_type:
        # 10 classes per task
        if data_percent == 2:
            return (80, 20, 1000)
        elif data_percent == 4:
            return (170, 30, 1000)
        elif data_percent == 6:
            return (250, 50, 1000)
        elif data_percent == 8:
            return (330, 70, 1000)
        elif data_percent == 10:
            return (410, 90, 1000)
        elif data_percent == 20:
            return (830, 170, 1000)
        elif data_percent == 25:
            return (1040, 210, 1000)
        elif data_percent == 30:
            return (1250, 250, 1000)
        elif data_percent == 40:
            return (1670, 330, 1000)
        elif data_percent == 50:
            return (2080, 420, 1000)
        elif data_percent == 60:
            return (2500, 500, 1000)
        elif data_percent == 70:
            return (2920, 580, 1000)
        elif data_percent == 75:
            return (3120, 630, 1000)
        elif data_percent == 80:
            return (3330, 670, 1000)
        elif data_percent == 90:
            return (3750, 750, 1000)
        elif data_percent == 100:
            return (4170, 830, 1000)
        else:
            return (None, None, None)
    elif 'omniglot' in data_type:
        if data_percent == 10:
            return (1, 1, 10)
        elif data_percent == 15:
            return (2, 1, 10)
        elif data_percent == 20:
            return (3, 1, 10)
        elif data_percent == 25:
            return (4, 1, 10)
        elif data_percent == 30:
            return (5, 1, 10)
        elif data_percent == 35:
            return (5, 2, 10)
        elif data_percent == 40:
            return (6, 2, 10)
        elif data_percent == 45:
            return (7, 2, 10)
        elif data_percent == 50:
            return (8, 2, 10)
        else:
            return (None, None, None)
    elif 'stl10' in data_type:
        if data_percent == 5:
            return (210, 40, 8000)
        elif data_percent == 10:
            return (415, 85, 8000)
        elif data_percent == 20:
            return (835, 165, 8000)
        elif data_percent == 30:
            return (1250, 250, 8000)
        elif data_percent == 40:
            return (1665, 335, 8000)
        elif data_percent == 50:
            return (2085, 415, 8000)
        elif data_percent == 60:
            return (2500, 500, 8000)
        elif data_percent == 70:
            return (2915, 585, 8000)
        elif data_percent == 80:
            return (3335, 665, 8000)
        elif data_percent == 90:
            return (3750, 750, 8000)
        else:
            return (None, None, None)

def model_setup(data_type, data_input_dim, model_type, test_type=0, cnn_padding_type_same=True, skip_connect_test_type=0, highway_connect_test_type=0, num_clayers=-1, phase1_max_epoch=100, darts_approx_order=1):
    model_architecture = None
    model_hyperpara = {}
    if cnn_padding_type_same:
        model_hyperpara['padding_type'] = 'SAME'
    else:
        model_hyperpara['padding_type'] = 'VALID'
    model_hyperpara['max_pooling'] = True
    model_hyperpara['dropout'] = True
    model_hyperpara['image_dimension'] = data_input_dim
    model_hyperpara['skip_connect'] = []
    model_hyperpara['highway_connect'] = highway_connect_test_type
    model_hyperpara['hidden_activation'] = relu

    if 'mnist' in data_type:
        model_hyperpara['batch_size'] = 10
        if 'ffnn' in model_type.lower():
            model_hyperpara['hidden_layer'] = [256, 128]
        else:
            ## CNN-FFNN case
            model_hyperpara['hidden_layer'] = [32]
        model_hyperpara['kernel_sizes'] = [5, 5, 5, 5]
        model_hyperpara['stride_sizes'] = [1, 1, 1, 1]
        model_hyperpara['channel_sizes'] = [32, 64]
        model_hyperpara['pooling_size'] = [2, 2, 2, 2]

        if skip_connect_test_type > 0:
            for _ in range(10):
                print("\tSkip connection isn't set up properly!!")
            #model_hyperpara['skip_connect']

    elif ('cifar10' in data_type) and not ('cifar100' in data_type):
        model_hyperpara['batch_size'] = 20
        if 'ffnn' in model_type.lower():
            model_hyperpara['hidden_layer'] = [256, 64]
        else:
            ## CNN-FFNN case
            model_hyperpara['hidden_layer'] = [64]
        model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3, 3, 3]
        model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1]
        model_hyperpara['channel_sizes'] = [32, 32, 64, 64]
        model_hyperpara['pooling_size'] = [1, 1, 2, 2, 1, 1, 2, 2]

        if skip_connect_test_type > 0:
            for _ in range(10):
                print("\tSkip connection isn't set up properly!!")
            #model_hyperpara['skip_connect']

    elif 'cifar100' in data_type:
        if num_clayers < 1:
            num_clayers = 4

        model_hyperpara['batch_size'] = 10
        if num_clayers == 4:
            model_hyperpara['hidden_layer'] = [64]
            model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 32, 64, 64]
            model_hyperpara['pooling_size'] = [1, 1, 2, 2, 1, 1, 2, 2]

            if skip_connect_test_type == 1:
                model_hyperpara['skip_connect'] = [[1, 1]]
            elif skip_connect_test_type == 2:
                model_hyperpara['skip_connect'] = [[3, 3]]
            elif skip_connect_test_type == 3:
                model_hyperpara['skip_connect'] = [[1, 1], [3, 3]]

        elif num_clayers == 6:
            model_hyperpara['hidden_layer'] = [64]
            model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 32, 64, 64, 128, 128]
            model_hyperpara['pooling_size'] = [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]

        elif num_clayers == 8:
            model_hyperpara['hidden_layer'] = [64]
            model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 32, 64, 64, 128, 128, 128, 128]
            model_hyperpara['pooling_size'] = [1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2, 1, 1, 2, 2]

        elif num_clayers == 17:
            ## ResNet-18 Architecture
            model_hyperpara['hidden_layer'] = [1000]
            model_hyperpara['kernel_sizes'] = [7, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [64, 64, 64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 512, 512, 512, 512]
            model_hyperpara['pooling_size'] = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 2, 1, 1, 1, 1, 1, 1]
            model_hyperpara['skip_connect'] = [[1, 2], [3, 4], [7, 8], [11, 12], [15, 16]]

        else:
            raise NotImplementedError

    elif 'omniglot' in data_type:
        model_hyperpara['batch_size'] = 10
        if 'ffnn' in model_type.lower():
            model_hyperpara['hidden_layer'] = [256, 128]
        else:
            ## CNN-FFNN case
            model_hyperpara['hidden_layer'] = [64]
        model_hyperpara['kernel_sizes'] = [5, 5, 3, 3, 3, 3]
        model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1]
        model_hyperpara['channel_sizes'] = [8, 16, 32]
        model_hyperpara['pooling_size'] = [2, 2, 2, 2, 2, 2]

        if skip_connect_test_type > 0:
            for _ in range(10):
                print("\tSkip connection isn't set up properly!!")
            #model_hyperpara['skip_connect']

    elif 'officehome' in data_type:
        if num_clayers < 1:
            num_clayers = 4

        model_hyperpara['batch_size'] = 16

        '''
        ## Original Office-Home paper's architecture
        model_hyperpara['hidden_layer'] = [4096, 1024, 64]
        model_hyperpara['kernel_sizes'] = [11, 11, 5, 5, 3, 3, 3, 3, 3, 3]
        model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        model_hyperpara['channel_sizes'] = [64, 256, 256, 256, 256]
        model_hyperpara['pooling_size'] = [3, 3, 3, 3, 1, 1, 1, 1, 3, 3]
        '''

        if num_clayers == 4:
            ## Hyper-parameter for IJCAI19 paper
            model_hyperpara['hidden_layer'] = [256, 64]
            model_hyperpara['kernel_sizes'] = [11, 11, 5, 5, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [64, 256, 256, 256]
            model_hyperpara['pooling_size'] = [3, 3, 3, 3, 2, 2, 2, 2]

        elif num_clayers == 6:
            ## Hyper-parameter for deeper net (ResNet or Highway Net)
            model_hyperpara['hidden_layer'] = [256, 64]
            model_hyperpara['kernel_sizes'] = [11, 11, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [64, 256, 256, 256, 256, 256]
            model_hyperpara['pooling_size'] = [3, 3, 3, 3, 1, 1, 2, 2, 1, 1, 2, 2]

            if skip_connect_test_type == 1:
                model_hyperpara['skip_connect'] = [[5, 5]]
            elif skip_connect_test_type == 2:
                model_hyperpara['skip_connect'] = [[2, 2], [3, 3], [4, 4]]
            elif skip_connect_test_type == 3:
                model_hyperpara['skip_connect'] = [[4, 4], [5, 5]]
            elif skip_connect_test_type == 4:
                model_hyperpara['skip_connect'] = [[2, 2], [3, 3]]
            elif skip_connect_test_type == 5:
                model_hyperpara['skip_connect'] = [[3, 3], [4, 4], [5, 5]]
            elif skip_connect_test_type == 6:
                model_hyperpara['skip_connect'] = [[2, 2]]
            elif skip_connect_test_type == 7:
                model_hyperpara['skip_connect'] = [[3, 3], [5, 5]]
            elif skip_connect_test_type == 8:
                model_hyperpara['skip_connect'] = [[2, 2], [4, 4]]
            elif skip_connect_test_type == 9:
                model_hyperpara['skip_connect'] = [[2,2], [3, 3], [4, 4], [5, 5]]

        elif num_clayers == 8:
            model_hyperpara['hidden_layer'] = [256, 64]
            model_hyperpara['kernel_sizes'] = [7, 7, 7, 7, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 32, 64, 64, 128, 128, 256, 256]
            model_hyperpara['pooling_size'] = [1, 1, 3, 3, 1, 1, 3, 3, 1, 1, 2, 2, 1, 1, 2, 2]

            if skip_connect_test_type == 1:
                model_hyperpara['skip_connect'] = [[1, 1], [3, 3], [5, 5], [7, 7]]

        else:
            raise NotImplementedError

    elif 'fashion' in data_type:
        raise NotImplementedError
        model_hyperpara['batch_size'] = 16
        if 'ffnn' in model_type.lower():
            model_hyperpara['hidden_layer'] = [128, 64, 32]
        else:
            ## CNN-FFNN case
            model_hyperpara['hidden_layer'] = [64]
        model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3]
        model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1]
        model_hyperpara['channel_sizes'] = [32, 64, 128]
        model_hyperpara['pooling_size'] = [1, 1, 2, 2, 2, 2]

        if skip_connect_test_type > 0:
            for _ in range(10):
                print("\tSkip connection isn't set up properly!!")
            #model_hyperpara['skip_connect']

    elif 'stl' in data_type or 'stl10' in data_type:
        model_hyperpara['batch_size'] = 10
        if 'ffnn' in model_type.lower():
            model_hyperpara['hidden_layer'] = [256, 64]
        else:
            ## CNN-FFNN case
            model_hyperpara['hidden_layer'] = [128, 16]

        if num_clayers == 6:
            model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 32, 64, 64, 128, 128]
            model_hyperpara['pooling_size'] = [1, 1, 3, 3, 1, 1, 3, 3, 1, 1, 3, 3]

        elif num_clayers == 9:
            model_hyperpara['kernel_sizes'] = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
            model_hyperpara['stride_sizes'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
            model_hyperpara['channel_sizes'] = [32, 32, 32, 64, 64, 64, 128, 128, 128]
            model_hyperpara['pooling_size'] = [1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3, 1, 1, 1, 1, 3, 3]

        else:
            raise NotImplementedError

        if skip_connect_test_type > 0:
            for _ in range(10):
                print("\tSkip connection isn't set up properly!!")
            #model_hyperpara['skip_connect']

    if model_type.lower() == 'stl':
        model_architecture = 'stl_cnn'

    elif model_type.lower() == 'stl_ffnn':
        model_architecture = 'stl_ffnn'

    elif model_type.lower() == 'snn':
        model_architecture = 'singlenn_cnn'

    elif model_type.lower() == 'snn_ffnn':
        model_architecture = 'singlenn_ffnn'

    elif model_type.lower() == 'hps' or model_type.lower() == 'hybrid_hps':
        model_architecture = 'hybrid_hps_cnn'

    elif model_type.lower() == 'hps_ffnn' or model_type.lower() == 'hybrid_hps_ffnn':
        model_architecture = 'hybrid_hps_ffnn'
        #tmp_layer_size = model_hyperpara['hidden_layer'].pop(-1)
        #model_hyperpara['task_specific_layer'] = [tmp_layer_size]
        model_hyperpara['task_specific_layer'] = []

    elif model_type.lower() == 'lasem_hps':
        model_architecture = 'lasem_hps_cnn'

    elif model_type.lower() == 'lasem_group_hps':
        model_architecture = 'lasemG_hps_cnn'

    elif model_type.lower() == 'darts_hps':
        model_architecture = 'darts_hps_cnn'
        model_hyperpara['darts_approx_order'] = darts_approx_order

    elif model_type.lower() == 'bruteforce_hps':
        model_architecture = 'bruteforce_hps_cnn'

    elif model_type.lower() == 'tf':
        model_architecture = 'mtl_tf_cnn'
        model_hyperpara['tensor_factor_type'] = 'Tucker'
        if test_type == 0:
            model_hyperpara['tensor_factor_error_threshold'] = 0.3
        elif test_type == 1:
            model_hyperpara['tensor_factor_error_threshold'] = 0.1
        elif test_type == 2:
            model_hyperpara['tensor_factor_error_threshold'] = 1e-3
        elif test_type == 3:
            model_hyperpara['tensor_factor_error_threshold'] = 1e-5
        elif test_type == 4:
            model_hyperpara['tensor_factor_error_threshold'] = 1e-7
        elif test_type == 5:
            model_hyperpara['tensor_factor_error_threshold'] = 1e-9

    elif model_type.lower() == 'tf_ffnn':
        model_architecture = 'mtl_tf_ffnn'
        #tmp_layer_size = model_hyperpara['hidden_layer'].pop(-1)
        #model_hyperpara['task_specific_layer'] = [tmp_layer_size]
        model_hyperpara['task_specific_layer'] = []
        model_hyperpara['tensor_factor_type'] = 'Tucker'
        if test_type == 0:
            model_hyperpara['tensor_factor_error_threshold'] = 0.3
        elif test_type == 1:
            model_hyperpara['tensor_factor_error_threshold'] = 0.1
        elif test_type == 2:
            model_hyperpara['tensor_factor_error_threshold'] = 1e-3
        elif test_type == 3:
            model_hyperpara['tensor_factor_error_threshold'] = 1e-5
        elif test_type == 4:
            model_hyperpara['tensor_factor_error_threshold'] = 1e-7
        elif test_type == 5:
            model_hyperpara['tensor_factor_error_threshold'] = 1e-9

    elif model_type.lower() == 'hybrid_tf' or model_type.lower() == 'lasem_tf' or model_type.lower() == 'lasem_group_tf' or model_type.lower() == 'bruteforce_tf':
        if 'em_group' in model_type.lower():
            model_architecture = 'lasemG_tf_cnn'
        elif 'em' in model_type.lower():
            model_architecture = 'lasem_tf_cnn'
        elif 'bruteforce' in model_type.lower():
            model_architecture = 'bruteforce_tf_cnn'
        else:
            model_architecture = 'hybrid_tf_cnn'

        if ('stl' in data_type or 'stl10' in data_type) and ('20t' in data_type):
            model_hyperpara['hidden_layer'] = [128]

        if test_type%5 == 0:
            model_hyperpara['auxiliary_loss_weight'] = 0.1
        elif test_type%5 == 1:
            model_hyperpara['auxiliary_loss_weight'] = 0.05
        elif test_type%5 == 2:
            model_hyperpara['auxiliary_loss_weight'] = 0.01
        elif test_type%5 == 3:
            model_hyperpara['auxiliary_loss_weight'] = 0.005
        elif test_type%5 == 4:
            model_hyperpara['auxiliary_loss_weight'] = 0.001

    elif model_type.lower() == 'prog' or model_type.lower() == 'prognn':
        model_architecture = 'prognn_cnn'
        if test_type == 0:
            model_hyperpara['dim_reduction_scale'] = 1.0
        elif test_type == 1:
            model_hyperpara['dim_reduction_scale'] = 2.0

    elif model_type.lower() == 'den':
        model_architecture = 'den_cnn'
        model_hyperpara['l1_lambda'] = 1e-6
        model_hyperpara['l2_lambda'] = 0.01
        model_hyperpara['gl_lambda'] = 0.8
        model_hyperpara['reg_lambda'] = 0.5
        model_hyperpara['loss_threshold'] = 0.01
        model_hyperpara['sparsity_threshold'] = [0.98, 0.01]

        if 'cifar' in data_type:
            if num_clayers == 4:
                model_hyperpara['den_expansion'] = 16
                model_hyperpara['hidden_layer'] = [128]
                model_hyperpara['channel_sizes'] = [32, 32, 64, 64]
        elif 'officehome' in data_type:
            if num_clayers == 4:
                model_hyperpara['channel_sizes'] = [64, 256, 256, 256]
                model_hyperpara['hidden_layer'] = [256, 64]
                model_hyperpara['den_expansion'] = 32

    elif model_type.lower() == 'ewc':
        model_architecture = 'ewc_cnn'

        if 'cifar100' in data_type:
            model_hyperpara['ewc_batch_size'] = 17
            if test_type == 0:
                model_hyperpara['fisher_multiplier'] = 0.001
            elif test_type == 1:
                model_hyperpara['fisher_multiplier'] = 1.0
            elif test_type == 2:
                model_hyperpara['fisher_multiplier'] = 1000.0
            elif test_type == 3:
                model_hyperpara['fisher_multiplier'] = 1e5
            elif test_type == 4:
                model_hyperpara['fisher_multiplier'] = 1e7
        elif 'officehome' in data_type:
            model_hyperpara['ewc_batch_size'] = 25
            if test_type == 0:
                model_hyperpara['fisher_multiplier'] = 0.001
            elif test_type == 1:
                model_hyperpara['fisher_multiplier'] = 1.0
            elif test_type == 2:
                model_hyperpara['fisher_multiplier'] = 1000.0
            elif test_type == 3:
                model_hyperpara['fisher_multiplier'] = 1e5
            elif test_type == 4:
                model_hyperpara['fisher_multiplier'] = 1e7

    elif model_type.lower() == 'channel_gate':
        model_architecture = 'channel_gated_cnn'
        model_hyperpara['sparsity_loss_weight'] = 100.0
        #model_hyperpara['gating_module_size'] = [256]
        #model_hyperpara['gating_temperature'] = 1.0/10.0
        model_hyperpara['gating_module_size'] = [256]
        model_hyperpara['gating_temperature'] = 1.0/10.0

    elif model_type.lower() == 'apd' or model_type.lower() == 'additive_param_decompose':
        model_architecture = 'additive_param_decompose_cnn'
        model_hyperpara['lambda1'] = 4e-4
        model_hyperpara['lambda2'] = 100

    elif model_type.lower() == 'dfcnn_direct':
        model_architecture = 'dfcnn_direct'

        if test_type%5 == 0:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-7]
        elif test_type%5 == 1:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-9]
        elif test_type%5 == 2:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-11]
        elif test_type%5 == 3:
            model_hyperpara['regularization_scale'] = [0.0, 1e-9, 0.0, 1e-9]
        elif test_type%5 == 4:
            model_hyperpara['regularization_scale'] = [0.0, 1e-9, 0.0, 1e-11]

        if data_type == 'mnist5':
            model_hyperpara['cnn_KB_sizes'] = [3, 24, 3, 48]
            model_hyperpara['cnn_TS_sizes'] = [3, 3]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
        elif data_type == 'mnist10':
            model_hyperpara['cnn_KB_sizes'] = [3, 24, 3, 64]
            model_hyperpara['cnn_TS_sizes'] = [3, 3]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
        elif data_type == 'cifar10_5':
            model_hyperpara['cnn_KB_sizes'] = [2, 24, 2, 48, 2, 64, 2, 72]
            model_hyperpara['cnn_TS_sizes'] = [3, 3, 3, 3]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-11]
        elif data_type == 'cifar10_10':
            model_hyperpara['cnn_KB_sizes'] = [2, 24, 2, 48, 2, 64, 2, 72]
            model_hyperpara['cnn_TS_sizes'] = [3, 3, 3, 3]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
        elif 'cifar100' in data_type:
            model_hyperpara['cnn_KB_sizes'] = [2, 24, 2, 48, 2, 64, 2, 72]
            model_hyperpara['cnn_TS_sizes'] = [3, 3, 3, 3]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
        elif 'omniglot' in data_type:
            model_hyperpara['cnn_KB_sizes'] = [3, 6, 2, 12, 2, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 3, 3]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2]

    elif (model_type.lower() == 'dfcnn') or (model_type.lower() == 'dfcnn_stl'):
        if model_type.lower() == 'dfcnn':
            model_architecture = 'dfcnn'
        else:
            model_architecture = 'dfcnn_singletask'

        if test_type%5 == 0:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-7]
        elif test_type%5 == 1:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-9]
        elif test_type%5 == 2:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-11]
        elif test_type%5 == 3:
            model_hyperpara['regularization_scale'] = [0.0, 1e-9, 0.0, 1e-9]
        elif test_type%5 == 4:
            model_hyperpara['regularization_scale'] = [0.0, 1e-9, 0.0, 1e-11]

        if data_type == 'mnist5':
            model_hyperpara['cnn_KB_sizes'] = [3, 12, 3, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
        elif data_type == 'mnist10':
            model_hyperpara['cnn_KB_sizes'] = [3, 12, 3, 40]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 64]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
        elif ('cifar10' in data_type) and not ('cifar100' in data_type):
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
            model_hyperpara['cnn_KB_sizes'] = [2, 16, 2, 24, 2, 32, 2, 36]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 64, 3, 72]
        elif 'cifar100' in data_type:
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
            model_hyperpara['cnn_KB_sizes'] = [2, 16, 2, 24, 2, 32, 2, 36]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 64, 3, 72]
        elif 'omniglot' in data_type:
            model_hyperpara['cnn_KB_sizes'] = [3, 3, 2, 8, 2, 16]
            model_hyperpara['cnn_TS_sizes'] = [3, 6, 3, 16, 3, 32]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2]
        elif 'officehome' in data_type:
            if num_clayers == 4:
                model_hyperpara['cnn_KB_sizes'] = [6, 24, 3, 64, 2, 64, 2, 64]
                model_hyperpara['cnn_TS_sizes'] = [3, 48, 3, 128, 3, 128, 3, 128]
                model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
            else:
                raise NotImplementedError

    elif (model_type.lower() == 'dfcnn_tc2') or (model_type.lower() == 'dfcnn_tc2_stl'):
        if model_type.lower() == 'dfcnn_tc2':
            model_architecture = 'dfcnn_tc2'
        else:
            model_architecture = 'dfcnn_tc2_singletask'

        if test_type%5 == 0:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-7]
        elif test_type%5 == 1:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-9]
        elif test_type%5 == 2:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-11]
        elif test_type%5 == 3:
            model_hyperpara['regularization_scale'] = [0.0, 1e-9, 0.0, 1e-9]
        elif test_type%5 == 4:
            model_hyperpara['regularization_scale'] = [0.0, 1e-9, 0.0, 1e-11]

        if data_type == 'mnist5':
            model_hyperpara['cnn_KB_sizes'] = [3, 1, 12, 3, 16, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
        elif data_type == 'mnist10':
            model_hyperpara['cnn_KB_sizes'] = [3, 1, 12, 3, 16, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2]
        elif data_type == 'cifar10_5':
            model_hyperpara['cnn_KB_sizes'] = [2, 2, 12, 2, 12, 12, 2, 12, 24, 2, 24, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 24, 3, 48, 3, 48]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
        elif data_type == 'cifar10_10':
            model_hyperpara['cnn_KB_sizes'] = [2, 2, 12, 2, 12, 12, 2, 12, 24, 2, 24, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 24, 3, 48, 3, 48]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
        elif 'cifar100' in data_type:
            model_hyperpara['cnn_KB_sizes'] = [2, 2, 12, 2, 12, 12, 2, 12, 24, 2, 24, 24]
            model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 24, 3, 48, 3, 48]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
        elif 'omniglot' in data_type:
            model_hyperpara['cnn_KB_sizes'] = [3, 1, 3, 2, 4, 6, 2, 12, 12]
            model_hyperpara['cnn_TS_sizes'] = [3, 6, 3, 12, 3, 24]
            model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2]

    elif (model_type.lower() == 'hybrid_dfcnn') or (model_type.lower() == 'darts_dfcnn') or ('dfcnn' in model_type.lower() and 'lasem' in model_type.lower()) or ('dfcnn' in model_type.lower() and 'bruteforce' in model_type.lower()):
        set_conv_sharing_bias = False
        if 'hybrid' in model_type.lower():
            model_architecture = 'hybrid_dfcnn'
        elif 'darts' in model_type.lower():
            model_architecture = 'darts_dfcnn'
            model_hyperpara['darts_approx_order'] = darts_approx_order
        elif 'em' in model_type.lower():
            if 'fixed' in model_type.lower():
                model_architecture = 'lasem_fixed_dfcnn'
            elif 'group' in model_type.lower():
                model_architecture = 'lasemG_dfcnn'
            else:
                model_architecture = 'lasem_dfcnn'
        elif 'bruteforce' in model_type.lower():
            model_architecture = 'bruteforce_dfcnn'
        else:
            raise ValueError

        if test_type%5 == 0:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-7]
        elif test_type%5 == 1:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-9]
        elif test_type%5 == 2:
            model_hyperpara['regularization_scale'] = [0.0, 1e-7, 0.0, 1e-11]
        elif test_type%5 == 3:
            model_hyperpara['regularization_scale'] = [0.0, 1e-9, 0.0, 1e-9]
        elif test_type%5 == 4:
            model_hyperpara['regularization_scale'] = [0.0, 1e-9, 0.0, 1e-11]

        if data_type == 'mnist5':
            assert ValueError, "Not yet modified!"
        elif data_type == 'mnist10':
            assert ValueError, "Not yet modified!"
        elif ('cifar10' in data_type) and not ('cifar100' in data_type):
            assert ValueError, "Not yet modified!"

        elif 'cifar100' in data_type:
            if num_clayers == 4:
                print("\nHybrid DF-CNN 4 layers\n")
                model_hyperpara['cnn_KB_sizes'] = [2, 16, 2, 24, 2, 32, 2, 36]
                model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 64, 3, 72]
                model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
                if set_conv_sharing_bias:
                    model_hyperpara['conv_sharing_bias'] = [0.0, 0.0, 0.0, 0.0]
            elif num_clayers == 6:
                print("\nHybrid DF-CNN 6 layers\n")
                model_hyperpara['cnn_KB_sizes'] = [2, 16, 2, 24, 2, 32, 2, 36, 2, 64, 2, 72]
                model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 64, 3, 72, 3, 128, 3, 144]
                model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                if set_conv_sharing_bias:
                    model_hyperpara['conv_sharing_bias'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif num_clayers == 8:
                print("\nHybrid DF-CNN 8 layers\n")
                model_hyperpara['cnn_KB_sizes'] = [2, 16, 2, 24, 2, 32, 2, 36, 2, 64, 2, 72, 2, 72, 2, 72]
                model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 64, 3, 72, 3, 128, 3, 144, 3, 144, 3, 144]
                model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                if set_conv_sharing_bias:
                    model_hyperpara['conv_sharing_bias'] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                raise NotImplementedError

        elif 'officehome' in data_type:
            if num_clayers == 4:
                print("\nHybrid DF-CNN 4 layers\n")
                model_hyperpara['cnn_KB_sizes'] = [6, 24, 3, 64, 2, 64, 2, 64]
                model_hyperpara['cnn_TS_sizes'] = [3, 48, 3, 128, 3, 128, 3, 128]
                model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2]
                if set_conv_sharing_bias:
                    model_hyperpara['conv_sharing_bias'] = [0.0, 0.0, 0.0, 0.0]
            elif num_clayers == 6:
                print("\nHybrid DF-CNN 6 layers\n")
                model_hyperpara['cnn_KB_sizes'] = [6, 24, 3, 64, 2, 72, 2, 72, 2, 72, 2, 72]
                model_hyperpara['cnn_TS_sizes'] = [5, 48, 3, 128, 3, 144, 3, 144, 3, 144, 3, 144]
                model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                if set_conv_sharing_bias:
                    model_hyperpara['conv_sharing_bias'] = [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0]
            elif num_clayers == 8:
                print("\nHybrid DF-CNN 8 layers\n")
                model_hyperpara['cnn_KB_sizes'] = [4, 12, 4, 24, 3, 32, 3, 48, 2, 64, 2, 72, 2, 72, 2, 72]
                model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 64, 3, 96, 3, 128, 3, 144, 3, 144, 3, 144]
                model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
                if set_conv_sharing_bias:
                    assert ValueError, "Need to recheck bias for conv sharing variables!"
                    model_hyperpara['conv_sharing_bias'] = [-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
            else:
                raise NotImplementedError
        elif 'stl10' in data_type:
            if num_clayers == 6:
                model_hyperpara['cnn_KB_sizes'] = [2, 12, 2, 24, 2, 32, 2, 48, 2, 64, 2, 96]
                model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 64, 3, 96, 3, 128, 3, 192]
                model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            elif num_clayers == 9:
                model_hyperpara['cnn_KB_sizes'] = [2, 12, 2, 24, 2, 24, 2, 32, 2, 48, 2, 48, 2, 64, 2, 96, 2, 96]
                model_hyperpara['cnn_TS_sizes'] = [3, 24, 3, 48, 3, 48, 3, 64, 3, 96, 3, 96, 3, 128, 3, 192, 3, 192]
                model_hyperpara['cnn_deconv_stride_sizes'] = [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
            else:
                raise NotImplementedError

    else:
        model_hyperpara = None

    if (model_type.lower() == 'hps') or (model_type.lower() == 'hybrid_hps') or (model_type.lower() == 'hybrid_tf') or (model_type.lower() == 'hybrid_dfcnn'):
        if num_clayers == 4:
            if test_type < 5:
                model_hyperpara['conv_sharing'] = [True, True, True, True]
            elif test_type < 10:
                model_hyperpara['conv_sharing'] = [False, False, False, True]
            elif test_type < 15:
                model_hyperpara['conv_sharing'] = [False, False, True, True]
            elif test_type < 20:
                model_hyperpara['conv_sharing'] = [False, True, True, True]
            elif test_type < 25:
                model_hyperpara['conv_sharing'] = [True, False, False, False]
            elif test_type < 30:
                model_hyperpara['conv_sharing'] = [True, True, False, False]
            elif test_type < 35:
                model_hyperpara['conv_sharing'] = [True, True, True, False]
            elif test_type < 40:
                model_hyperpara['conv_sharing'] = [False, True, False, True]
            elif test_type < 45:
                model_hyperpara['conv_sharing'] = [True, False, True, False]

        elif num_clayers == 6:
            if test_type < 5:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, True]
            elif test_type < 10:
                model_hyperpara['conv_sharing'] = [False, False, False, False, False, True]
            elif test_type < 15:
                model_hyperpara['conv_sharing'] = [False, False, False, False, True, True]
            elif test_type < 20:
                model_hyperpara['conv_sharing'] = [False, False, False, True, True, True]
            elif test_type < 25:
                model_hyperpara['conv_sharing'] = [False, False, True, True, True, True]
            elif test_type < 30:
                model_hyperpara['conv_sharing'] = [False, True, True, True, True, True]
            elif test_type < 35:
                model_hyperpara['conv_sharing'] = [True, False, False, False, False, False]
            elif test_type < 40:
                model_hyperpara['conv_sharing'] = [True, True, False, False, False, False]
            elif test_type < 45:
                model_hyperpara['conv_sharing'] = [True, True, True, False, False, False]
            elif test_type < 50:
                model_hyperpara['conv_sharing'] = [True, True, True, True, False, False]
            elif test_type < 55:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, False]
            elif test_type < 60:
                model_hyperpara['conv_sharing'] = [False, True, False, True, False, True]

        elif num_clayers == 8:
            if test_type < 5:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, True, True, True]
            elif test_type < 10:
                model_hyperpara['conv_sharing'] = [False, False, False, False, False, False, True, True]
            elif test_type < 15:
                model_hyperpara['conv_sharing'] = [False, False, False, False, True, True, True, True]
            elif test_type < 20:
                model_hyperpara['conv_sharing'] = [False, False, True, True, True, True, True, True]
            elif test_type < 25:
                model_hyperpara['conv_sharing'] = [True, True, False, False, False, False, False, False]
            elif test_type < 30:
                model_hyperpara['conv_sharing'] = [True, True, True, True, False, False, False, False]
            elif test_type < 35:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, True, False, False]
            elif test_type < 40:
                model_hyperpara['conv_sharing'] = [False, True, False, True, False, True, False, True]
            elif test_type < 45:
                model_hyperpara['conv_sharing'] = [True, False, True, False, True, False, True, False]

        elif num_clayers == 9:
            if test_type < 5:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, True, True, True, True]

        elif num_clayers == 17:
            if test_type < 5:
                model_hyperpara['conv_sharing'] = [True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]

        else:
            raise NotImplementedError

        if 'officehome' in data_type and num_clayers == 6:
            if test_type < 5:
                model_hyperpara['conv_sharing'] = [False, False, True, True, True, True]
            elif test_type < 10:
                model_hyperpara['conv_sharing'] = [False, False, False, False, False, True]
            elif test_type < 15:
                model_hyperpara['conv_sharing'] = [False, False, False, False, True, True]
            elif test_type < 20:
                model_hyperpara['conv_sharing'] = [False, False, False, True, True, True]
            elif test_type < 25:
                model_hyperpara['conv_sharing'] = [False, False, True, False, False, False]
            elif test_type < 30:
                model_hyperpara['conv_sharing'] = [False, False, True, True, False, False]
            elif test_type < 35:
                model_hyperpara['conv_sharing'] = [False, False, True, True, True, False]
            elif test_type < 40:
                model_hyperpara['conv_sharing'] = [False, False, False, True, False, True]
            elif test_type < 45:
                model_hyperpara['conv_sharing'] = [False, False, True, False, True, False]

    if 'lasemG' in model_architecture:
        if num_clayers == 9:
            model_hyperpara['layer_group_config'] = [3, 3, 3]
        elif num_clayers == 17:
            if test_type < 5:
                model_hyperpara['layer_group_config'] = [5, 4, 4, 4]
            elif test_type < 10:
                model_hyperpara['layer_group_config'] = [5, 4, 4, 2, 2]
            elif test_type < 15:
                model_hyperpara['layer_group_config'] = [5, 4, 2, 2, 2, 2]
            elif test_type < 20:
                model_hyperpara['layer_group_config'] = [5, 2, 2, 2, 2, 2, 2]
            elif test_type < 25:
                model_hyperpara['layer_group_config'] = [3, 2, 2, 2, 2, 2, 2, 2]
        elif num_clayers == 4:
            model_hyperpara['layer_group_config'] = [1, 1, 1, 1]

        else:
            raise NotImplementedError

    return (model_architecture, model_hyperpara)
