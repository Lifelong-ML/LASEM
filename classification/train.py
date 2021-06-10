import os
import timeit
from time import sleep
from random import shuffle

import numpy as np
import tensorflow as tf
from scipy.io import savemat

from classification.gen_data import mnist_data_print_info, cifar_data_print_info, officehome_data_print_info, print_data_info
from utils.utils import savemat_wrapper, data_augmentation_STL_analysis, convert_dataset_to_oneHot, data_augmentation_in_minibatch

from classification.model.cnn_baseline_model import LL_several_CNN_minibatch, LL_single_CNN_minibatch, LL_CNN_HPS_minibatch, LL_CNN_progressive_net, LL_CNN_tensorfactor_minibatch, LL_CNN_ChannelGatedNet, LL_CNN_APD
from classification.model.cnn_den_model import CNN_FC_DEN
from classification.model.cnn_dfcnn_model import LL_hybrid_DFCNN_minibatch
from classification.model.cnn_darts_model import LL_HPS_CNN_DARTS_net, LL_DFCNN_DARTS_net
from classification.model.cnn_lasem_model import LL_CNN_HPS_EM_algo, LL_hybrid_TF_EM_algo, LL_hybrid_DFCNN_EM_algo

_tf_ver = tf.__version__.split('.')
_up_to_date_tf = int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) > 14)

_debug_mode = False

#### function to generate appropriate deep neural network
def model_generation(model_architecture, model_hyperpara, train_hyperpara, data_info, classification_prob=False, data_list=None, tfInitParam=None, lifelong=False):
    learning_model, gen_model_success = None, True
    learning_rate = train_hyperpara['lr']
    learning_rate_decay = train_hyperpara['lr_decay']

    if len(data_info) == 3:
        x_dim, y_dim, y_depth = data_info
    elif len(data_info) == 4:
        x_dim, y_dim, y_depth, num_task = data_info

    if lifelong:
        fc_hidden_sizes = list(model_hyperpara['hidden_layer'])
    else:
        if isinstance(y_depth, list) or type(y_depth) == np.ndarray:
            fc_hidden_sizes = [list(model_hyperpara['hidden_layer'])+[y_d] for y_d in y_depth]
        else:
            fc_hidden_size = model_hyperpara['hidden_layer'] + [y_depth]
            fc_hidden_sizes = [fc_hidden_size for _ in range(num_task)]

    cnn_kernel_size, cnn_kernel_stride, cnn_channel_size = model_hyperpara['kernel_sizes'], model_hyperpara['stride_sizes'], model_hyperpara['channel_sizes']
    cnn_padding, cnn_pooling, cnn_dropout = model_hyperpara['padding_type'], model_hyperpara['max_pooling'], model_hyperpara['dropout']
    if cnn_pooling:
        cnn_pool_size = model_hyperpara['pooling_size']
    else:
        cnn_pool_size = None

    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']
    if 'regularization_scale' in model_hyperpara:
        regularization_scale = model_hyperpara['regularization_scale']

    input_size = model_hyperpara['image_dimension']
    skip_connect = model_hyperpara['skip_connect']
    highway_connect = model_hyperpara['highway_connect']

    ###### CNN models
    if model_architecture == 'stl_cnn':
        if lifelong:
            print("Training STL-CNNs model (collection of NN per a task) - Lifelong Learning")
            learning_model = LL_several_CNN_minibatch(model_hyperpara, train_hyperpara)
        else:
            print("Need Lifelong Learning!!")
            raise NotImplementedError

    elif model_architecture == 'singlenn_cnn':
        if lifelong:
            print("Training a single CNN model for all tasks - Lifelong Learning")
            learning_model =  LL_single_CNN_minibatch(model_hyperpara, train_hyperpara)
        else:
            print("Need Lifelong Learning!!")
            raise NotImplementedError

    elif model_architecture == 'hybrid_hps_cnn':
        if lifelong:
            print("Training HPS-CNNs model (Hard-parameter Sharing) - Lifelong Learning")
            learning_model = LL_CNN_HPS_minibatch(model_hyperpara, train_hyperpara)
        else:
            print("Need Lifelong Learning!!")
            raise NotImplementedError
        print("\tConfig of sharing: ", model_hyperpara['conv_sharing'])

    elif model_architecture == 'mtl_tf_cnn' or model_architecture == 'hybrid_tf_cnn':
        if lifelong:
            print("Training LL-CNN model (Tensorfactorization ver.)")
            learning_model = LL_CNN_tensorfactor_minibatch(model_hyperpara, train_hyperpara)
            print("\tConfig of sharing: ", model_hyperpara['conv_sharing'])
        else:
            print("Need Lifelong Learning!!")
            raise NotImplementedError

    elif model_architecture == 'prognn_cnn':
        print("Training LL-CNN Progressive model")
        if lifelong:
            learning_model = LL_CNN_progressive_net(model_hyperpara, train_hyperpara)
        else:
            print("Progressive Neural Net requires 'lifelong learning' mode!")
            raise NotImplementedError

    elif model_architecture == 'den_cnn':
        print("Training LL-CNN Dynamically Expandable model")
        if lifelong:
            learning_model = CNN_FC_DEN(model_hyperpara, train_hyperpara, data_info)
        else:
            print("Dynamically Expandable Net requires 'lifelong learning' mode!")
            raise NotImplementedError

    elif ('channel_gated' in model_architecture):
        print("Training Conditional Channel-gated Network model")
        if lifelong:
            learning_model = LL_CNN_ChannelGatedNet(model_hyperpara, train_hyperpara)
        else:
            print("Need Lifelong Learning!!")
            raise NotImplementedError

    elif ('apd' in model_architecture or 'additive_param' in model_architecture):
        print("Training Additive Parameter Decomposition (APD) model")
        if lifelong:
            learning_model = LL_CNN_APD(model_hyperpara, train_hyperpara)
        else:
            print("Need Lifelong Learning!!")
            raise NotImplementedError

    elif model_architecture == 'hybrid_dfcnn':
        cnn_know_base_size, cnn_task_specific_size, cnn_deconv_stride_size = model_hyperpara['cnn_KB_sizes'], model_hyperpara['cnn_TS_sizes'], model_hyperpara['cnn_deconv_stride_sizes']
        cnn_sharing = model_hyperpara['conv_sharing']
        if lifelong:
            print("Training Hybrid DF-CNNs model (ResNet skip-conn.) - Lifelong Learning")
            learning_model = LL_hybrid_DFCNN_minibatch(model_hyperpara, train_hyperpara)
        else:
            print("Need Lifelong Learning!!")
            raise NotImplementedError
        print("\tConfig of sharing: ", cnn_sharing)

    elif model_architecture == 'lasem_hps_cnn' or model_architecture == 'lasemG_hps_cnn':
        if lifelong:
            print("Training Hybrid HPS-CNNs model (Hard-parameter Sharing/EM) - Lifelong Learning")
            learning_model = LL_CNN_HPS_EM_algo(model_hyperpara, train_hyperpara)
        else:
            print("Training Hybrid HPS-CNNs model (Hard-parameter Sharing/EM) - Multi-task Learning")
            raise NotImplementedError
        if 'lasemG' in model_architecture:
            print("\tGroups of layers to select: ", model_hyperpara['layer_group_config'])
    elif model_architecture == 'lasem_tf_cnn' or model_architecture == 'lasemG_tf_cnn':
        if lifelong:
            print("Training Hybrid TF-CNNs model (Tensor-factorized Sharing/EM) - Lifelong Learning")
            learning_model = LL_hybrid_TF_EM_algo(model_hyperpara, train_hyperpara)
        else:
            print("Training Hybrid TF-CNNs model (Tensor-factorized Sharing/EM) - Multi-task Learning")
            raise NotImplementedError
        if 'lasemG' in model_architecture:
            print("\tGroups of layers to select: ", model_hyperpara['layer_group_config'])
    elif ('lasem' in model_architecture) and ('dfcnn' in model_architecture):
        if lifelong:
            print("Training Hybrid DF-CNNs model (ResNet skip-conn./auto sharing/EM) - Lifelong Learning")
            learning_model = LL_hybrid_DFCNN_EM_algo(model_hyperpara, train_hyperpara)
        else:
            print("Training Hybrid DF-CNNs model (ResNet skip-conn./auto sharing/EM) - Multi-task Learning")
            raise NotImplementedError
        if 'lasemG' in model_architecture:
            print("\tGroups of layers to select: ", model_hyperpara['layer_group_config'])

    elif model_architecture == 'darts_hps_cnn':
        if lifelong:
            print("Training Hybrid HPS-CNNs model (Hard-parameter Sharing/DARTS) - Lifelong Learning")
            learning_model = LL_HPS_CNN_DARTS_net(model_hyperpara, train_hyperpara)
        else:
            print("Need Lifelong Learning!!")
            raise NotImplementedError
    elif model_architecture == 'darts_dfcnn':
        if lifelong:
            print("Training Hybrid DF-CNN model (DF-CNN/DARTS) - Lifelong Learning")
            learning_model = LL_DFCNN_DARTS_net(model_hyperpara, train_hyperpara)
        else:
            print("Need Lifelong Learning!!")
            raise NotImplementedError

    else:
        print("No such model exists!!")
        print("No such model exists!!")
        print("No such model exists!!")
        gen_model_success = False

    if learning_model is not None:
        learning_model.model_architecture = model_architecture
    sleep(5)
    return (learning_model, gen_model_success)


#### module of training/testing one model
def train_mtl(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, classification_prob, useGPU=False, GPU_device=0, save_param=False, param_folder_path='saved_param', save_param_interval=100, save_graph=False, tfInitParam=None, run_cnt=0):
    print("Training function for multi-task learning (NOT lifelong learning)!")
    assert ('progressive' not in model_architecture and 'den' not in model_architecture and 'dynamically' not in model_architecture), "Use train function appropriate to the architecture"

    ### control log of TensorFlow
    #os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
    #tf.logging.set_verbosity(tf.logging.ERROR)

    config = tf.ConfigProto()
    if useGPU:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(GPU_device)
        if _up_to_date_tf:
            ## TF version > 1.14
            gpu = tf.config.experimental.list_physical_devices('GPU')[0]
            tf.config.experimental.set_memory_growth(gpu, True)
        else:
            ## TF version <= 1.14
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.9
        print("GPU %d is used" %(GPU_device))
    else:
        os.environ["CUDA_VISIBLE_DEVICES"]=""
        print("CPU is used")


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


    ### Set hyperparameter related to training process
    learning_step_max = train_hyperpara['learning_step_max']
    improvement_threshold = train_hyperpara['improvement_threshold']
    patience = train_hyperpara['patience']
    patience_multiplier = train_hyperpara['patience_multiplier']
    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']


    ### Generate Model
    learning_model, generation_success = model_generation(model_architecture, model_hyperpara, train_hyperpara, [x_dim, y_dim, y_depth, num_task], classification_prob=classification_prob, data_list=dataset, tfInitParam=tfInitParam, lifelong=False)
    if not generation_success:
        return (None, None)

    ### Training Procedure
    learning_step = -1
    if (('batch_size' in locals()) or ('batch_size' in globals())) and (('num_task' in locals()) or ('num_task' in globals())):
        if num_task > 1:
            indices = [list(range(num_train[x])) for x in range(num_task)]
        else:
            #indices = [range(num_train)]
            indices = [list(range(num_train[0]))]

    best_valid_error, test_error_at_best_epoch, best_epoch, epoch_bias = np.inf, np.inf, -1, 0
    train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = [], [], [], []

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if save_graph:
            tfboard_writer = tf.summary.FileWriter('./graphs/%s/run%d'%(model_architecture, run_cnt), sess.graph)

        start_time = timeit.default_timer()
        while learning_step < min(learning_step_max, epoch_bias + patience):
            learning_step = learning_step+1

            #### training & performance measuring process
            task_for_train = np.random.randint(0, num_task)

            if classification_prob:
                model_train_error, model_valid_error, model_test_error = learning_model.train_accuracy, learning_model.valid_accuracy, learning_model.test_accuracy
            else:
                model_train_error, model_valid_error, model_test_error = learning_model.train_loss, learning_model.valid_loss, learning_model.test_loss

            if learning_step > 0:
                shuffle(indices[task_for_train])

                for batch_cnt in range(num_train[task_for_train]//batch_size):
                    batch_train_x = train_data[task_for_train][0][indices[task_for_train][batch_cnt*batch_size:(batch_cnt+1)*batch_size], :]
                    batch_train_y = train_data[task_for_train][1][indices[task_for_train][batch_cnt*batch_size:(batch_cnt+1)*batch_size]]

                    if train_hyperpara['data_augment']:
                        # data_augmentation_in_minibatch(data_x, data_y, image_dimension)
                        batch_train_x, batch_train_y = data_augmentation_in_minibatch(batch_train_x, batch_train_y, model_hyperpara['image_dimension'])

                    if ('cnn' in model_architecture):
                        sess.run(learning_model.update[task_for_train], feed_dict={learning_model.model_input[task_for_train]: batch_train_x, learning_model.true_output[task_for_train]: batch_train_y, learning_model.epoch: learning_step-1, learning_model.dropout_prob: 0.5})
                    else:
                        sess.run(learning_model.update[task_for_train], feed_dict={learning_model.model_input[task_for_train]: batch_train_x, learning_model.true_output[task_for_train]: batch_train_y, learning_model.epoch: learning_step-1})

            train_error_tmp = [0.0 for _ in range(num_task)]
            validation_error_tmp = [0.0 for _ in range(num_task)]
            test_error_tmp = [0.0 for _ in range(num_task)]
            for task_cnt in range(num_task):
                for batch_cnt in range(num_train[task_cnt]//batch_size):
                    if ('cnn' in model_architecture):
                        train_error_tmp[task_cnt] = train_error_tmp[task_cnt] + sess.run(model_train_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: train_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: train_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size], learning_model.dropout_prob: 1.0})
                    else:
                        train_error_tmp[task_cnt] = train_error_tmp[task_cnt] + sess.run(model_train_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: train_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: train_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size]})
                train_error_tmp[task_cnt] = train_error_tmp[task_cnt]/((num_train[task_cnt]//batch_size)*batch_size)

                for batch_cnt in range(num_valid[task_cnt]//batch_size):
                    if ('cnn' in model_architecture):
                        validation_error_tmp[task_cnt] = validation_error_tmp[task_cnt] + sess.run(model_valid_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: validation_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: validation_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size], learning_model.dropout_prob: 1.0})
                    else:
                        validation_error_tmp[task_cnt] = validation_error_tmp[task_cnt] + sess.run(model_valid_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: validation_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: validation_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size]})
                validation_error_tmp[task_cnt] = validation_error_tmp[task_cnt]/((num_valid[task_cnt]//batch_size)*batch_size)

                for batch_cnt in range(num_test[task_cnt]//batch_size):
                    if ('cnn' in model_architecture):
                        test_error_tmp[task_cnt] = test_error_tmp[task_cnt] + sess.run(model_test_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: test_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: test_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size], learning_model.dropout_prob: 1.0})
                    else:
                        test_error_tmp[task_cnt] = test_error_tmp[task_cnt] + sess.run(model_test_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: test_data[task_cnt][0][batch_cnt * batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: test_data[task_cnt][1][batch_cnt * batch_size:(batch_cnt+1)*batch_size]})
                test_error_tmp[task_cnt] = test_error_tmp[task_cnt]/((num_test[task_cnt]//batch_size)*batch_size)

                if classification_prob:
                    ## for classification, error_tmp is actually ACCURACY, thus, change the sign for checking improvement
                    train_error, valid_error, test_error = -(sum(train_error_tmp)/num_task), -(sum(validation_error_tmp)/num_task), -(sum(test_error_tmp)/num_task)
                else:
                    train_error, valid_error, test_error = np.sqrt(np.array(train_error_tmp)/num_task), np.sqrt(np.array(validation_error_tmp)/num_task), np.sqrt(np.array(test_error_tmp)/num_task)
                    train_error_tmp, validation_error_tmp, test_error_tmp = list(np.sqrt(np.array(train_error_tmp))), list(np.sqrt(np.array(validation_error_tmp))), list(np.sqrt(np.array(test_error_tmp)))
                train_error_to_compare, valid_error_to_compare, test_error_to_compare = train_error, valid_error, test_error

            #### error related process
            print('epoch %d - Train : %f, Validation : %f' % (learning_step, abs(train_error_to_compare), abs(valid_error_to_compare)))

            if valid_error_to_compare < best_valid_error:
                str_temp = ''
                if valid_error_to_compare < best_valid_error * improvement_threshold:
                    patience = max(patience, (learning_step-epoch_bias)*patience_multiplier)
                    str_temp = '\t<<'
                best_valid_error, best_epoch = valid_error_to_compare, learning_step
                test_error_at_best_epoch = test_error_to_compare
                print('\t\t\t\t\t\t\tTest : %f%s' % (abs(test_error_at_best_epoch), str_temp))

            train_error_hist.append(train_error_tmp + [abs(train_error)])
            valid_error_hist.append(validation_error_tmp + [abs(valid_error)])
            test_error_hist.append(test_error_tmp + [abs(test_error)])
            best_test_error_hist.append(abs(test_error_at_best_epoch))

        if save_param:
            para_file_name = param_folder_path + '/final_model_parameter.mat'
            curr_param = learning_model.get_params_val(sess)
            savemat(para_file_name, {'parameter': curr_param})

    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" %(end_time-start_time))
    print("Best validation error : %.4f (at epoch %d)" %(abs(best_valid_error), best_epoch))
    print("Test error at that epoch (%d) : %.4f" %(best_epoch, abs(test_error_at_best_epoch)))

    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['num_epoch'] = learning_step
    result_summary['best_epoch'] = best_epoch
    result_summary['history_train_error'] = train_error_hist
    result_summary['history_validation_error'] = valid_error_hist
    result_summary['history_test_error'] = test_error_hist
    result_summary['history_best_test_error'] = best_test_error_hist
    result_summary['best_validation_error'] = abs(best_valid_error)
    result_summary['test_error_at_best_epoch'] = abs(test_error_at_best_epoch)

    if save_graph:
        tfboard_writer.close()

    return result_summary, learning_model.num_trainable_var



#### module of training/testing one model
def train_lifelong(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, classification_prob, useGPU=False, GPU_device=0, save_param=False, param_folder_path='saved_param', save_param_interval=100, save_graph=False, tfInitParam=None, run_cnt=0):
    print("Training function for lifelong learning!")
    assert ('den' not in model_architecture and 'dynamically' not in model_architecture), "Use train function appropriate to the architecture"

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


    if train_hyperpara['stl_analysis']:
        print("\nAnalysis on STL according to the amount of training data")
        print("\tTask : %d, Ratio : %.2f" %(train_hyperpara['stl_task_to_learn'], train_hyperpara['stl_total_data_ratio']))
        train_data, validation_data, test_data = data_augmentation_STL_analysis(dataset, train_hyperpara['stl_task_to_learn'], train_hyperpara['stl_total_data_ratio'], model_hyperpara['image_dimension'])
        task_training_order = [train_hyperpara['stl_task_to_learn']]
    else:
        if 'task_order' not in train_hyperpara.keys():
            task_training_order = list(range(train_hyperpara['num_tasks']))
        else:
            task_training_order = list(train_hyperpara['task_order'])
        #for cnt in range(20):
        #    print("This is only for debugging!!!!!")
        #task_training_order = [8, 5, 8, 2, 3, 5, 9, 0, 1, 2, 4, 6, 8, 7]
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
    else:
        raise ValueError("No specified dataset!")

    if train_hyperpara['stl_analysis']:
        print("New number of train data:")
        print(num_train)
        print("\n")


    ### Set hyperparameter related to training process
    learning_step_max = train_hyperpara['learning_step_max']
    improvement_threshold = train_hyperpara['improvement_threshold']
    patience = train_hyperpara['patience']
    patience_multiplier = train_hyperpara['patience_multiplier']
    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']


    ### Generate Model
    learning_model, generation_success = model_generation(model_architecture, model_hyperpara, train_hyperpara, [x_dim, y_dim, y_depth, num_task], classification_prob=classification_prob, data_list=dataset, tfInitParam=tfInitParam, lifelong=True)
    if not generation_success:
        return (None, None)

    ### Training Procedure
    learning_step = -1
    if num_task > 1:
        indices = [list(range(num_train[x])) for x in range(num_task)]
    else:
        indices = [list(range(num_train[0]))]

    best_valid_error, test_error_at_best_epoch, best_epoch, epoch_bias = np.inf, np.inf, -1, 0
    train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = [], [], [], []

    transfer_score_onTrain = np.zeros([len(task_training_order), num_task])
    transfer_score_onValid = np.zeros([len(task_training_order), num_task])
    transfer_score_onTest = np.zeros([len(task_training_order), num_task])

    start_time = timeit.default_timer()
    for train_task_cnt, (task_for_train) in enumerate(task_training_order):
        tf.reset_default_graph()

        with tf.Session(config=config) as sess:
            print("\nTask - %d"%(task_for_train))
            learning_model.add_new_task(y_depth[task_for_train], task_for_train, single_input_placeholder=True)
            num_learned_tasks = learning_model.number_of_learned_tasks()

            sess.run(tf.global_variables_initializer())
            if save_graph:
                tfboard_writer = tf.summary.FileWriter('./graphs/%s/run%d/task%d'%(model_architecture, run_cnt, train_task_cnt), sess.graph)
            #opts = tf.profiler.ProfileOptionBuilder.float_operation()
            #flops = tf.profiler.profile(sess.graph, run_meta=tf.RunMetadata(), cmd='op', options=opts)
            #print("\tFLOPS : %d"%(flops))

            if save_param and _debug_mode:
                para_file_name = param_folder_path + '/init_model_parameter_taskC%d.mat'%(train_task_cnt)
                curr_param = learning_model.get_params_val(sess)
                savemat(para_file_name, {'parameter': curr_param})

            ## Compute LEEP transferability score
            if num_learned_tasks > 1 and train_hyperpara['LEEP_score']:
                for tmp_cnt, (task_index_to_eval) in enumerate(task_training_order[:train_task_cnt]):
                    if task_index_to_eval in task_training_order[:tmp_cnt] or task_index_to_eval==task_for_train:
                        continue
                    transfer_score_onTrain[train_task_cnt, task_index_to_eval] = learning_model.compute_transferability_score_one_task(sess, train_data[task_for_train][0], train_data[task_for_train][1], task_index_to_eval)
                    transfer_score_onValid[train_task_cnt, task_index_to_eval] = learning_model.compute_transferability_score_one_task(sess, validation_data[task_for_train][0], validation_data[task_for_train][1], task_index_to_eval)
                    transfer_score_onTest[train_task_cnt, task_index_to_eval] = learning_model.compute_transferability_score_one_task(sess, test_data[task_for_train][0], test_data[task_for_train][1], task_index_to_eval)

            while learning_step < min(learning_step_max, epoch_bias + patience):
                learning_step = learning_step+1

                #### training & performance measuring process
                if learning_step > 0:
                    learning_model.train_one_epoch(sess, train_data[task_for_train][0], train_data[task_for_train][1], learning_step-1, task_for_train, indices[task_for_train], dropout_prob=0.5)

                train_error_tmp = [0.0 for _ in range(num_task)]
                validation_error_tmp = [0.0 for _ in range(num_task)]
                test_error_tmp = [0.0 for _ in range(num_task)]
                for tmp_cnt, (task_index_to_eval) in enumerate(task_training_order[:train_task_cnt+1]):
                    if task_index_to_eval in task_training_order[:tmp_cnt]:
                        continue
                    train_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, train_data[task_index_to_eval][0], train_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)
                    validation_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, validation_data[task_index_to_eval][0], validation_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)
                    test_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, test_data[task_index_to_eval][0], test_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)

                if classification_prob:
                    ## for classification, error_tmp is actually ACCURACY, thus, change the sign for checking improvement
                    train_error, valid_error, test_error = -(sum(train_error_tmp)/(num_learned_tasks)), -(sum(validation_error_tmp)/(num_learned_tasks)), -(sum(test_error_tmp)/(num_learned_tasks))
                    train_error_to_compare, valid_error_to_compare, test_error_to_compare = -train_error_tmp[task_for_train], -validation_error_tmp[task_for_train], -test_error_tmp[task_for_train]
                else:
                    train_error, valid_error, test_error = np.sqrt(np.array(train_error_tmp)/(num_learned_tasks)), np.sqrt(np.array(validation_error_tmp)/(num_learned_tasks)), np.sqrt(np.array(test_error_tmp)/(num_learned_tasks))
                    train_error_tmp, validation_error_tmp, test_error_tmp = list(np.sqrt(np.array(train_error_tmp))), list(np.sqrt(np.array(validation_error_tmp))), list(np.sqrt(np.array(test_error_tmp)))
                    train_error_to_compare, valid_error_to_compare, test_error_to_compare = train_error_tmp[task_for_train], validation_error_tmp[task_for_train], test_error_tmp[task_for_train]

                #### error related process
                print('epoch %d - Train : %f, Validation : %f' % (learning_step, abs(train_error_to_compare), abs(valid_error_to_compare)))

                if valid_error_to_compare < best_valid_error:
                    str_temp = ''
                    if valid_error_to_compare < best_valid_error * improvement_threshold:
                        patience = max(patience, (learning_step-epoch_bias)*patience_multiplier)
                        str_temp = '\t<<'
                    best_valid_error, best_epoch = valid_error_to_compare, learning_step
                    test_error_at_best_epoch = test_error_to_compare
                    print('\t\t\t\t\t\t\tTest : %f%s' % (abs(test_error_at_best_epoch), str_temp))

                train_error_hist.append(train_error_tmp + [abs(train_error)])
                valid_error_hist.append(validation_error_tmp + [abs(valid_error)])
                test_error_hist.append(test_error_tmp + [abs(test_error)])
                best_test_error_hist.append(abs(test_error_at_best_epoch))

                #if learning_step >= epoch_bias+min(patience, learning_step_max//num_task):
                if learning_step >= epoch_bias+min(patience, learning_step_max//len(task_training_order)):
                    if save_param:
                        para_file_name = param_folder_path + '/model_parameter_taskC%d_task%d.mat'%(train_task_cnt, task_for_train)
                        curr_param = learning_model.get_params_val(sess)
                        savemat(para_file_name, {'parameter': curr_param})

                    if train_task_cnt == len(task_training_order)-1:
                        if save_param:
                            para_file_name = param_folder_path + '/final_model_parameter.mat'
                            curr_param = learning_model.get_params_val(sess)
                            savemat(para_file_name, {'parameter': curr_param})
                    else:
                        # update epoch_bias, task_for_train, task_change_epoch
                        epoch_bias = learning_step
                        task_change_epoch.append(learning_step+1)

                        # initialize best_valid_error, best_epoch, patience
                        patience = train_hyperpara['patience']
                        best_valid_error, best_epoch = np.inf, -1

                        learning_model.convert_tfVar_to_npVar(sess)
                        print('\n\t>>Change to new task!<<\n')
                    break

    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" %(end_time-start_time))

    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['num_epoch'] = learning_step
    result_summary['history_train_error'] = train_error_hist
    result_summary['history_validation_error'] = valid_error_hist
    result_summary['history_test_error'] = test_error_hist
    result_summary['history_best_test_error'] = best_test_error_hist

    tmp_valid_error_hist = np.array(valid_error_hist)
    chk_epoch = [(task_change_epoch[x], task_change_epoch[x+1]) for x in range(len(task_change_epoch)-1)] + [(task_change_epoch[-1], learning_step+1)]
    #tmp_best_valid_error_list = [np.amax(tmp_valid_error_hist[x[0]:x[1], t]) for x, t in zip(chk_epoch, range(num_task))]
    #result_summary['best_validation_error'] = sum(tmp_best_valid_error_list) / float(len(tmp_best_valid_error_list))
    result_summary['task_changed_epoch'] = task_change_epoch

    #if model_architecture == 'hybrid_dfcnn_auto_sharing':
    if 'hybrid_dfcnn_auto_sharing' in model_architecture:
        result_summary['conv_sharing'] = learning_model.conv_sharing

    ## LEEP transferability score
    result_summary['LEEP_score_trainset'] = transfer_score_onTrain
    result_summary['LEEP_score_validset'] = transfer_score_onValid
    result_summary['LEEP_score_testset'] = transfer_score_onTest

    if save_graph:
        tfboard_writer.close()

    return result_summary, learning_model.num_trainable_var



#### module of training/testing one model
def train_den_net(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, classification_prob, doLifelong, useGPU=False, GPU_device=0, save_param=False, param_folder_path='saved_param', save_param_interval=100, save_graph=False):
    assert (('den' in model_architecture or 'dynamically' in model_architecture) and classification_prob and doLifelong), "Use train function appropriate to the architecture (Dynamically Expandable Net)"

    print("\nTrain function for Dynamically Expandable Net is called!\n")

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
        print("CPU is used")

    ### set-up data
    train_data, validation_data, test_data = dataset
    if 'mnist' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = mnist_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif 'cifar10' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif data_type == 'cifar100':
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = cifar_data_print_info(train_data, validation_data, test_data, True, print_info=False)
    elif 'officehome' in data_type:
        num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth = officehome_data_print_info(train_data, validation_data, test_data, True, print_info=False)


    ### reformat data for DEN
    trainX, trainY = [train_data[t][0] for t in range(num_task)], [train_data[t][1] for t in range(num_task)]
    validX, validY = [validation_data[t][0] for t in range(num_task)], [validation_data[t][1] for t in range(num_task)]
    testX, testY = [test_data[t][0] for t in range(num_task)], [test_data[t][1] for t in range(num_task)]


    if save_graph:
        if 'graphs' not in os.listdir(os.getcwd()):
            os.mkdir(os.getcwd()+'/graphs')


    ### Set hyperparameter related to training process
    learning_step_max = train_hyperpara['learning_step_max']
    improvement_threshold = train_hyperpara['improvement_threshold']
    patience = train_hyperpara['patience']
    patience_multiplier = train_hyperpara['patience_multiplier']
    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']


    ### Generate Model
    learning_model, generation_success = model_generation(model_architecture, model_hyperpara, train_hyperpara, [x_dim, y_dim, y_depth, num_task], classification_prob=classification_prob, data_list=dataset, lifelong=True)
    if not generation_success:
        return (None, None)

    learning_model.set_sess_config(config)

    params = dict()
    train_accuracy, valid_accuracy, test_accuracy, best_test_accuracy = [], [], [], []
    start_time = timeit.default_timer()
    for train_task_cnt in range(num_task):
        print("\n\nStart training new task %d" %(train_task_cnt))
        data = (trainX, trainY, validX, validY, testX, testY)

        learning_model.sess = tf.Session(config=config)

        learning_model.T = learning_model.T + 1
        learning_model.task_indices.append(train_task_cnt+1)
        learning_model.load_params(params, time = 1)
        tr_acc, v_acc, te_acc, best_te_acc = learning_model.add_task(train_task_cnt+1, data, save_param, save_graph)
        train_accuracy = train_accuracy+tr_acc
        valid_accuracy = valid_accuracy+v_acc
        test_accuracy = test_accuracy+te_acc
        best_test_accuracy = best_test_accuracy+best_te_acc

        params = learning_model.get_params()

        learning_model.destroy_graph()
        learning_model.sess.close()

    num_trainable_var = learning_model.num_trainable_var(params_list=params)
    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" %(end_time-start_time))

    task_change_epoch = learning_model.task_change_epoch

    tmp_valid_acc_hist = np.array(valid_accuracy)
    chk_epoch = [(task_change_epoch[x], task_change_epoch[x+1]) for x in range(len(task_change_epoch)-1)] # + [(task_change_epoch[-1], learning_step+1)]
    tmp_best_valid_acc_list = [np.amax(tmp_valid_acc_hist[x[0]:x[1], t]) for x, t in zip(chk_epoch, range(num_task))]

    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['num_epoch'] = learning_model.num_training_epoch
    result_summary['history_train_error'] = np.array(train_accuracy)
    result_summary['history_validation_error'] = np.array(valid_accuracy)
    result_summary['history_test_error'] = np.array(test_accuracy)
    result_summary['history_best_test_error'] = np.array(best_test_accuracy)
    result_summary['best_validation_error'] = sum(tmp_best_valid_acc_list) / float(len(tmp_best_valid_acc_list))
    result_summary['test_error_at_best_epoch'] = 0.0
    result_summary['task_changed_epoch'] = task_change_epoch[:-1]
    return result_summary, num_trainable_var


#### module of training/testing one model
def train_ewc(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, classification_prob, doLifelong, useGPU=False, GPU_device=0, save_param=False, param_folder_path='saved_param', save_param_interval=100, save_graph=False):
    assert ( ('ewc' not in model_architecture or 'elastic' not in model_architecture) and doLifelong ), "Use train function appropriate to the architecture"

    print("\nTrain function for EWC is called!\n")

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


    #task_training_order = list([0, 5, 1, 6, 2, 7, 3, 8, 4, 9]) #OfficeHome_new_order0
    #task_training_order = list([0, 5, 3, 8, 2, 7, 1, 6, 4, 9]) #OfficeHome_new_order1
    task_training_order = list(range(train_hyperpara['num_tasks']))
    task_for_train, task_change_epoch = task_training_order.pop(0), [1]


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


    ### reformat data for EWC (one-hot encoding)
    train_data = convert_dataset_to_oneHot(train_data, y_depth)
    validation_data = convert_dataset_to_oneHot(validation_data, y_depth)
    test_data = convert_dataset_to_oneHot(test_data, y_depth)


    ### Set hyperparameter related to training process
    learning_step_max = train_hyperpara['learning_step_max']
    improvement_threshold = train_hyperpara['improvement_threshold']
    patience = train_hyperpara['patience']
    patience_multiplier = train_hyperpara['patience_multiplier']
    if 'batch_size' in model_hyperpara:
        batch_size = model_hyperpara['batch_size']

    ### Generate Model
    learning_model, generation_success = model_generation(model_architecture, model_hyperpara, train_hyperpara, [x_dim, y_dim, y_depth, num_task], classification_prob=classification_prob, data_list=dataset)
    if not generation_success:
        return (None, None)

    ### Training Procedure
    best_param = []
    if save_param:
        best_para_file_name = param_folder_path+'/best_model_parameter'
        print("Saving trained parameters at '%s'" %(param_folder_path) )
    else:
        print("Not saving trained parameters")

    learning_step = -1
    if (('batch_size' in locals()) or ('batch_size' in globals())) and (('num_task' in locals()) or ('num_task' in globals())):
        if num_task > 1:
            indices = [list(range(num_train[x])) for x in range(num_task)]
        else:
            #indices = [range(num_train)]
            indices = [list(range(num_train[0]))]

    best_valid_error, test_error_at_best_epoch, best_epoch, epoch_bias = np.inf, np.inf, -1, 0
    train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = [], [], [], []

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        if save_graph:
            tfboard_writer = tf.summary.FileWriter('./graphs', sess.graph)

        model_train_error, model_valid_error, model_test_error = learning_model.train_accuracy, learning_model.valid_accuracy, learning_model.test_accuracy
        update_func = learning_model.update[task_for_train]

        start_time = timeit.default_timer()
        while learning_step < min(learning_step_max, epoch_bias + patience):
            learning_step = learning_step+1

            if learning_step > 1:
                learning_model.update_fisher_full_batch(sess, train_data[task_for_train][0], train_data[task_for_train][1])

            if learning_step > 0:
                shuffle(indices[task_for_train])
                for batch_cnt in range(num_train[task_for_train]//batch_size):
                    sess.run(update_func, feed_dict={learning_model.model_input[task_for_train]: train_data[task_for_train][0][indices[task_for_train][batch_cnt*batch_size:(batch_cnt+1)*batch_size], :], learning_model.true_output[task_for_train]: train_data[task_for_train][1][indices[task_for_train][batch_cnt*batch_size:(batch_cnt+1)*batch_size], :], learning_model.epoch: learning_step-1, learning_model.dropout_prob: 0.5})

            train_error_tmp = [0.0 for _ in range(num_task)]
            validation_error_tmp = [0.0 for _ in range(num_task)]
            test_error_tmp = [0.0 for _ in range(num_task)]
            for task_cnt in range(num_task):
                for batch_cnt in range(num_train[task_cnt]//batch_size):
                    train_error_tmp[task_cnt] = train_error_tmp[task_cnt] + sess.run(model_train_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: train_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: train_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.dropout_prob: 1.0})
                train_error_tmp[task_cnt] = train_error_tmp[task_cnt]/((num_train[task_cnt]//batch_size)*batch_size)

                for batch_cnt in range(num_valid[task_cnt]//batch_size):
                    validation_error_tmp[task_cnt] = validation_error_tmp[task_cnt] + sess.run(model_valid_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: validation_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: validation_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.dropout_prob: 1.0})
                validation_error_tmp[task_cnt] = validation_error_tmp[task_cnt]/((num_valid[task_cnt]//batch_size)*batch_size)

                for batch_cnt in range(num_test[task_cnt]//batch_size):
                    test_error_tmp[task_cnt] = test_error_tmp[task_cnt] + sess.run(model_test_error[task_cnt], feed_dict={learning_model.model_input[task_cnt]: test_data[task_cnt][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.true_output[task_cnt]: test_data[task_cnt][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], learning_model.dropout_prob: 1.0})
                test_error_tmp[task_cnt] = test_error_tmp[task_cnt]/((num_test[task_cnt]//batch_size)*batch_size)

            train_error, valid_error, test_error = -(sum(train_error_tmp)/num_task), -(sum(validation_error_tmp)/num_task), -(sum(test_error_tmp)/num_task)
            train_error_to_compare, valid_error_to_compare, test_error_to_compare = -train_error_tmp[task_for_train], -validation_error_tmp[task_for_train], -test_error_tmp[task_for_train]

            #### error related process
            print('epoch %d - Train : %f, Validation : %f' % (learning_step, abs(train_error_to_compare), abs(valid_error_to_compare)))

            if valid_error_to_compare < best_valid_error:
                str_temp = ''
                if valid_error_to_compare < best_valid_error * improvement_threshold:
                    patience = max(patience, (learning_step-epoch_bias)*patience_multiplier)
                    str_temp = '\t<<'
                best_valid_error, best_epoch = valid_error_to_compare, learning_step
                test_error_at_best_epoch = test_error_to_compare
                print('\t\t\t\t\t\t\tTest : %f%s' % (abs(test_error_at_best_epoch), str_temp))

                #### save best parameter of model
                if save_param:
                    best_param = sess.run(learning_model.param)
                    savemat(best_para_file_name + '.mat', {'parameter': savemat_wrapper(best_param)})

            #### save intermediate result of training procedure
            if (learning_step % save_param_interval == 0) and save_param:
                curr_param = sess.run(learning_model.param)
                para_file_name = param_folder_path + '/model_parameter(epoch_' + str(learning_step) + ')'
                savemat(para_file_name + '.mat', {'parameter': savemat_wrapper(curr_param)})

            if learning_step >= epoch_bias+min(patience, learning_step_max//num_task) and len(task_training_order) > 0:
                print('\n\t>>Change to new task!<<\n')
                epoch_bias, task_for_train = learning_step, task_training_order.pop(0)
                task_change_epoch.append(learning_step+1)
                update_func = learning_model.update_ewc[task_for_train]

                # update optimal parameter holder
                learning_model.update_lagged_param(sess)

                # initialize best_valid_error, best_epoch, patience
                patience = train_hyperpara['patience']
                best_valid_error, best_epoch = np.inf, -1

            train_error_hist.append(train_error_tmp + [abs(train_error)])
            valid_error_hist.append(validation_error_tmp + [abs(valid_error)])
            test_error_hist.append(test_error_tmp + [abs(test_error)])
            best_test_error_hist.append(abs(test_error_at_best_epoch))

        task_change_epoch.append(learning_step+1)

    end_time = timeit.default_timer()
    print("End of Training")
    print("Time consumption for training : %.2f" %(end_time-start_time))

    result_summary = {}
    result_summary['training_time'] = end_time - start_time
    result_summary['num_epoch'] = learning_step
    result_summary['best_epoch'] = best_epoch
    result_summary['history_train_error'] = train_error_hist
    result_summary['history_validation_error'] = valid_error_hist
    result_summary['history_test_error'] = test_error_hist
    result_summary['history_best_test_error'] = best_test_error_hist
    result_summary['best_validation_error'] = abs(best_valid_error)
    result_summary['test_error_at_best_epoch'] = abs(test_error_at_best_epoch)

    tmp_valid_error_hist = np.array(valid_error_hist)
    chk_epoch = [(task_change_epoch[x], task_change_epoch[x+1]) for x in range(len(task_change_epoch)-1)] + [(task_change_epoch[-1], learning_step+1)]
    tmp_best_valid_error_list = [np.amax(tmp_valid_error_hist[x[0]:x[1], t]) for x, t in zip(chk_epoch, range(num_task))]
    result_summary['best_validation_error'] = sum(tmp_best_valid_error_list) / float(len(tmp_best_valid_error_list))
    result_summary['task_changed_epoch'] = task_change_epoch

    if save_graph:
        tfboard_writer.close()

    return result_summary, learning_model.num_trainable_var
