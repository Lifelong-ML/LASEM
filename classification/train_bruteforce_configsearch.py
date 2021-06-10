import os
import timeit
from time import sleep
from random import shuffle

import numpy as np
import tensorflow as tf
from scipy.io import savemat

from classification.gen_data import mnist_data_print_info, cifar_data_print_info, officehome_data_print_info, print_data_info
from classification.model.cnn_bruteforce_config_search_model import LL_CNN_HPS_BruteForceSearch, LL_CNN_TF_BruteForceSearch, LL_CNN_DFCNN_BruteForceSearch

_tf_ver = tf.__version__.split('.')
_up_to_date_tf = int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) > 14)

#### function to generate appropriate deep neural network
def model_generation(model_architecture, model_hyperpara, train_hyperpara):
    learning_model = None

    ###### CNN models
    if 'hps' in model_architecture and 'bruteforce' in model_architecture:
        print("Training HPS-CNNs model (Hard-parameter Sharing) - Lifelong/Brute-force Config Search")
        learning_model = LL_CNN_HPS_BruteForceSearch(model_hyperpara, train_hyperpara, model_architecture)

    elif 'tensorfactor' in model_architecture and 'bruteforce' in model_architecture:
        print("Training TF-CNNs model (Tensor-Factorization) - Lifelong/Brute-force Config Search")
        learning_model = LL_CNN_TF_BruteForceSearch(model_hyperpara, train_hyperpara, model_architecture)

    elif 'dfcnn' in model_architecture and 'bruteforce' in model_architecture:
        print("Training DF-CNNs model - Lifelong/Brute-force Config Search")
        learning_model = LL_CNN_DFCNN_BruteForceSearch(model_hyperpara, train_hyperpara, model_architecture)

    else:
        raise ValueError

    sleep(5)
    return learning_model

def train_lifelong_bruteforce_config_search(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, useGPU=False, GPU_device=0, save_param=True, param_folder_path='saved_param', save_graph=False, run_cnt=0):
    print("Training function for brute-force transfer configuration search!")
    assert ('bruteforce' in model_architecture), "Use train function appropriate to the architecture"

    ### control log of TensorFlow
    #os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    #tf.logging.set_verbosity(tf.logging.ERROR)

    config = tf.ConfigProto()
    if useGPU:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_device)
        if _up_to_date_tf:
            ## TF version >= 1.14
            gpu = tf.config.experimental.list_physical_devices('GPU')[0]
            tf.config.experimental.set_memory_growth(gpu, True)
        else:
            ## TF version < 1.14
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
        print("GPU %d is used" %(GPU_device))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        print("CPU is used")


    if 'task_order' not in train_hyperpara.keys():
        task_training_order = list(range(train_hyperpara['num_tasks']))
    else:
        task_training_order = list(train_hyperpara['task_order'])
    task_change_epoch = [1]


    ### set-up data
    train_data, validation_data, test_data = dataset
    if 'mnist' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = mnist_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif ('cifar10' in data_type) and not ('cifar100' in data_type):
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif 'cifar100' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif 'officehome' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = officehome_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif 'stl10' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = print_data_info(train_data, validation_data, test_data, print_info=False)

    ### Generate Model
    learning_model = model_generation(model_architecture, model_hyperpara, train_hyperpara)
    if learning_model is None:
        return (None, None)

    learning_model.update_param_dir(param_folder_path)
    learning_model.update_tfsession_config(config)

    ### Training Procedure
    train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = None, None, None, None

    start_time = timeit.default_timer()
    for train_task_cnt, (task_for_train) in enumerate(task_training_order):
        print("\nTask - %d"%(task_for_train))
        task_train_error_hist, task_valid_error_hist, task_test_error_hist, task_best_test_error_hist = learning_model.train_a_task(y_depth[task_for_train], task_for_train, train_hyperpara['patience'], train_data, validation_data, test_data, save_graph=False, epoch_bias=train_task_cnt*train_hyperpara['patience'])

        if train_error_hist is None:
            train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = np.array(task_train_error_hist), np.array(task_valid_error_hist), np.array(task_test_error_hist), np.array(task_best_test_error_hist)
        else:
            train_error_hist = np.concatenate((train_error_hist, task_train_error_hist), axis=0)
            valid_error_hist = np.concatenate((valid_error_hist, task_valid_error_hist), axis=0)
            test_error_hist = np.concatenate((test_error_hist, task_test_error_hist), axis=0)
            best_test_error_hist = np.concatenate((best_test_error_hist, task_best_test_error_hist))
        task_change_epoch.append(task_change_epoch[-1]+train_hyperpara['patience'])

    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" %(end_time-start_time))
    _ = task_change_epoch.pop()

    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['num_epoch'] = len(train_error_hist)-1
    result_summary['history_train_error'] = train_error_hist
    result_summary['history_validation_error'] = valid_error_hist
    result_summary['history_test_error'] = test_error_hist
    result_summary['history_best_test_error'] = best_test_error_hist
    result_summary['conv_sharing'] = learning_model.conv_sharing
    result_summary['task_changed_epoch'] = task_change_epoch

    return result_summary, learning_model.num_trainable_var
