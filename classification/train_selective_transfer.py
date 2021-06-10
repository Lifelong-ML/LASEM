import os
import timeit
from random import shuffle

import numpy as np
import tensorflow as tf
from scipy.io import savemat

from classification.train import model_generation
from classification.gen_data import mnist_data_print_info, cifar_data_print_info, officehome_data_print_info, print_data_info

_tf_ver = tf.__version__.split('.')
_up_to_date_tf = int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) > 14)

_phase1_epoch_max = 100


###################################################################################################################
###################################################################################################################
######## Learn which layers to share based on EM algorithm
###################################################################################################################
###################################################################################################################
#### module of training/testing one model
def train_lifelong_LASEM(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, classification_prob, useGPU=False, GPU_device=0, save_param=False, param_folder_path='saved_param', save_param_interval=100, save_graph=False, tfInitParam=None, run_cnt=0):
    print("Training function for lifelong learning/Hybrid model with automatic sharing(EM)!")
    assert (model_architecture == 'lasem_hps_cnn' or model_architecture == 'lasemG_hps_cnn' or model_architecture == 'lasem_tf_cnn' or model_architecture == 'lasemG_tf_cnn' or model_architecture == 'lasem_dfcnn' or model_architecture == 'lasemG_dfcnn'), "Use train function appropriate to the architecture"

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
        raise ValueError

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
        return (None, None, None, None)

    ### Training Procedure
    best_param = []
    if save_param:
        best_para_file_name = param_folder_path+'/best_model_parameter'
        print("Saving trained parameters at '%s'" %(param_folder_path) )
    else:
        print("Not saving trained parameters")

    learning_step = -1
    if num_task > 1:
        indices = [list(range(num_train[x])) for x in range(num_task)]
    else:
        indices = [list(range(num_train[0]))]

    best_valid_error, test_error_at_best_epoch, best_epoch, epoch_bias = np.inf, np.inf, -1, 0
    train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = [], [], [], []
    config_curr_posterior_hist, config_prior_hist = [], []

    start_time = timeit.default_timer()
    for train_task_cnt, (task_for_train) in enumerate(task_training_order):
        print("\nTask - %d"%(task_for_train))
        tf.reset_default_graph()
        with tf.Session(config=config) as sess:
            learning_model.add_new_task(y_depth[task_for_train], task_for_train)
            task_model_index = learning_model.find_task_model(task_for_train)
            num_learned_tasks = learning_model.number_of_learned_tasks()

            sess.run(tf.global_variables_initializer())
            if save_graph:
                tfboard_writer = tf.summary.FileWriter('./graphs/%s/run%d/task%d'%(model_architecture, run_cnt, train_task_cnt), sess.graph)

            if not train_hyperpara['reset_prior']:
                if train_hyperpara['em_cnt_prior']:
                    accumulated_posterior_cnt = train_hyperpara['em_prior_init_cnt']*np.ones(len(learning_model._possible_configs), dtype=np.float32)
                else:
                    accumulated_posterior, den_acc_posterior = np.zeros(len(learning_model._possible_configs), dtype=np.float32), 0.0

            while learning_step < min(learning_step_max, epoch_bias + patience):
                learning_step = learning_step+1

                #### training
                best_config = 0
                if learning_step > 0:
                    shuffle(indices[task_for_train])

                    accumulated_posterior_cnt_inEpoch = train_hyperpara['em_prior_init_cnt']*np.ones(len(learning_model._possible_configs), dtype=np.float32)
                    accumulated_posterior_inEpoch, den_acc_posterior_inEpoch = np.zeros(len(learning_model._possible_configs), dtype=np.float32), 0.0
                    if train_hyperpara['reset_prior']:
                        if train_hyperpara['em_cnt_prior']:
                            sess.run(learning_model.update_prior, feed_dict={learning_model.posterior_placeholder: accumulated_posterior_cnt_inEpoch/np.sum(accumulated_posterior_cnt_inEpoch)})
                        else:
                            sess.run(learning_model.update_prior, feed_dict={learning_model.posterior_placeholder: np.ones(len(learning_model._possible_configs), dtype=np.float32)/float(len(learning_model._possible_configs))})
                    for batch_cnt in range(num_train[task_for_train]//batch_size):
                        batch_train_x = train_data[task_for_train][0][indices[task_for_train][batch_cnt*batch_size:(batch_cnt+1)*batch_size], :]
                        batch_train_y = train_data[task_for_train][1][indices[task_for_train][batch_cnt*batch_size:(batch_cnt+1)*batch_size]]

                        if learning_model.task_is_new:
                            config_posterior = sess.run(learning_model.posterior, feed_dict={learning_model.model_input[task_model_index]: batch_train_x, learning_model.true_output[task_model_index]: batch_train_y, learning_model.dropout_prob: 1.0, learning_model.num_data_in_batch: batch_size})

                            ## run update methods of configurations separately
                            #for up in learning_model.update:
                            #    sess.run(up, feed_dict={learning_model.model_input[task_model_index]: batch_train_x, learning_model.true_output[task_model_index]: batch_train_y, learning_model.epoch: learning_step-1, learning_model.dropout_prob: 0.5, learning_model.posterior_placeholder: config_posterior})

                            ## run update methods simultaneously (by adding grads)
                            sess.run(learning_model.update, feed_dict={learning_model.model_input[task_model_index]: batch_train_x, learning_model.true_output[task_model_index]: batch_train_y, learning_model.epoch: learning_step-1, learning_model.dropout_prob: 0.5, learning_model.posterior_placeholder: config_posterior})

                            if train_hyperpara['em_cnt_prior']:
                                max_config_minibatch = np.argmax(config_posterior)
                                accumulated_posterior_cnt_inEpoch[max_config_minibatch] += 1.0
                                accumulated_posterior_cnt[max_config_minibatch] += 1.0
                                if train_hyperpara['reset_prior']:
                                    sess.run(learning_model.update_prior, feed_dict={learning_model.posterior_placeholder: accumulated_posterior_cnt_inEpoch/np.sum(accumulated_posterior_cnt_inEpoch)})
                                else:
                                    sess.run(learning_model.update_prior, feed_dict={learning_model.posterior_placeholder: accumulated_posterior_cnt/np.sum(accumulated_posterior_cnt)})
                            else:
                                accumulated_posterior, accumulated_posterior_inEpoch = accumulated_posterior+config_posterior, accumulated_posterior_inEpoch+config_posterior
                                den_acc_posterior, den_acc_posterior_inEpoch = den_acc_posterior+1.0, den_acc_posterior_inEpoch+1.0
                                if train_hyperpara['reset_prior']:
                                    sess.run(learning_model.update_prior, feed_dict={learning_model.posterior_placeholder: accumulated_posterior_inEpoch*(1.0/den_acc_posterior_inEpoch)})
                                else:
                                    sess.run(learning_model.update_prior, feed_dict={learning_model.posterior_placeholder: accumulated_posterior*(1.0/den_acc_posterior)})
                        else:
                            sess.run(learning_model.update, feed_dict={learning_model.model_input[task_model_index]: batch_train_x, learning_model.true_output[task_model_index]: batch_train_y, learning_model.epoch: learning_step-1, learning_model.dropout_prob: 0.5})
                    if learning_model.task_is_new:
                        #accumulated_posterior *= (1.0/float(num_train[task_for_train]//batch_size))
                        #sess.run(learning_model.update_prior, feed_dict={learning_model.posterior_placeholder: accumulated_posterior})
                        best_config = learning_model.best_config(sess)
                    if train_hyperpara['em_cnt_prior']:
                        config_prior_hist.append(accumulated_posterior_cnt/np.sum(accumulated_posterior_cnt))
                        config_curr_posterior_hist.append(accumulated_posterior_cnt_inEpoch/np.sum(accumulated_posterior_cnt_inEpoch))
                    else:
                        config_prior_hist.append(accumulated_posterior*(1.0/den_acc_posterior))
                        config_curr_posterior_hist.append(accumulated_posterior_inEpoch*(1.0/den_acc_posterior_inEpoch))

                #### performance evaluation
                train_error_tmp = [0.0 for _ in range(num_task)]
                validation_error_tmp = [0.0 for _ in range(num_task)]
                test_error_tmp = [0.0 for _ in range(num_task)]
                for tmp_cnt, (task_index_to_eval) in enumerate(task_training_order[:train_task_cnt+1]):
                    if task_index_to_eval in task_training_order[:tmp_cnt]:
                        continue
                    train_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, train_data[task_index_to_eval][0], train_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)
                    validation_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, validation_data[task_index_to_eval][0], validation_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)
                    test_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, test_data[task_index_to_eval][0], test_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)

                ## for classification, error_tmp is actually ACCURACY, thus, change the sign for checking improvement
                train_error, valid_error, test_error = -(sum(train_error_tmp)/(num_learned_tasks)), -(sum(validation_error_tmp)/(num_learned_tasks)), -(sum(test_error_tmp)/(num_learned_tasks))
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

                train_error_hist.append(train_error_tmp + [abs(train_error)])
                valid_error_hist.append(validation_error_tmp + [abs(valid_error)])
                test_error_hist.append(test_error_tmp + [abs(test_error)])
                best_test_error_hist.append(abs(test_error_at_best_epoch))

                #if learning_step >= epoch_bias+min(patience, learning_step_max//num_task):
                if learning_step >= epoch_bias+min(patience, learning_step_max//len(task_training_order)):
                    learning_model.convert_tfVar_to_npVar(sess)

                    if save_param:
                        para_file_name = param_folder_path + '/model_parameter_taskC%d_task%d.mat'%(train_task_cnt, task_for_train)
                        curr_param = learning_model.get_params_val(sess, use_npparams=True)
                        savemat(para_file_name, {'parameter': curr_param})

                    if train_task_cnt == len(task_training_order)-1:
                        ## After training phase, compute mutual information
                        if save_param:
                            para_file_name = param_folder_path + '/final_model_parameter.mat'
                            curr_param = learning_model.get_params_val(sess, use_npparams=True)
                            savemat(para_file_name, {'parameter': curr_param})
                    else:
                        # update epoch_bias, task_for_train, task_change_epoch
                        epoch_bias = learning_step
                        task_change_epoch.append(learning_step+1)

                        # initialize best_valid_error, best_epoch, patience
                        patience = train_hyperpara['patience']
                        best_valid_error, best_epoch = np.inf, -1

                        print('\n\t>>Change to new task!<<\n')
                        if learning_model.task_is_new:
                            print('\t', learning_model._possible_configs[best_config])
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
    result_summary['history_net_prior_in_epoch'] = config_curr_posterior_hist
    result_summary['history_net_prior'] = config_prior_hist
    result_summary['conv_sharing'] = learning_model.conv_sharing

    tmp_valid_error_hist = np.array(valid_error_hist)
    chk_epoch = [(task_change_epoch[x], task_change_epoch[x+1]) for x in range(len(task_change_epoch)-1)] + [(task_change_epoch[-1], learning_step+1)]
    #tmp_best_valid_error_list = [np.amax(tmp_valid_error_hist[x[0]:x[1], t]) for x, t in zip(chk_epoch, range(num_task))]
    #result_summary['best_validation_error'] = sum(tmp_best_valid_error_list) / float(len(tmp_best_valid_error_list))
    result_summary['task_changed_epoch'] = task_change_epoch

    if save_graph:
        tfboard_writer.close()

    return result_summary, learning_model.num_trainable_var




########################################################
### functions to fix prior/posterior of specific configuration to see the effect of EM learning
########################################################
def gen_fixed_prob(list_of_configs, max_prob_config, prob_of_that_config):
    if max_prob_config == 'top1':
        config_in_bool = [False, False, False, True]
    elif max_prob_config == 'top2':
        config_in_bool = [False, False, True, True]
    elif max_prob_config == 'top3':
        config_in_bool = [False, True, True, True]
    elif max_prob_config == 'bottom1':
        config_in_bool = [True, False, False, False]
    elif max_prob_config == 'bottom2':
        config_in_bool = [True, True, False, False]
    elif max_prob_config == 'bottom3':
        config_in_bool = [True, True, True, False]
    elif max_prob_config == 'alt':
        config_in_bool = [False, True, False, True]

    tmp = [ all([A==B for (A, B) in zip(a, config_in_bool)]) for a in list_of_configs ]
    index_of_config, num_all_config = tmp.index(True), len(tmp)
    a, b = np.ones(num_all_config, dtype=np.float32)*(1.0-prob_of_that_config)/(num_all_config-1.0), np.zeros(num_all_config, dtype=np.float32)
    b[index_of_config] = prob_of_that_config - (1.0-prob_of_that_config)/(num_all_config-1.0)
    return a+b



#### module of training/testing one model
def train_lifelong_LASEM_fixedProb(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, classification_prob, useGPU=False, GPU_device=0, save_param=False, param_folder_path='saved_param', save_param_interval=100, save_graph=False, tfInitParam=None, run_cnt=0):
    print("Training function to analyze lifelong learning/Hybrid DF-CNN with automatic sharing!")
    print("\tFixed probability is given to all possible configurations")
    print("\tConfiguration: ", train_hyperpara['em_analysis_maxC'], "\tprob of max config: ", train_hyperpara['em_analysis_maxC_prob'])
    assert (model_architecture == 'lasem_fixed_dfcnn'), "Use train function appropriate to the architecture"

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
        raise ValueError

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
        return (None, None, None, None)

    ### Training Procedure
    best_param = []
    if save_param:
        best_para_file_name = param_folder_path+'/best_model_parameter'
        print("Saving trained parameters at '%s'" %(param_folder_path) )
    else:
        print("Not saving trained parameters")

    learning_step = -1
    if num_task > 1:
        indices = [list(range(num_train[x])) for x in range(num_task)]
    else:
        indices = [list(range(num_train[0]))]

    max_prob_config, prob_of_that_config = train_hyperpara['em_analysis_maxC'], train_hyperpara['em_analysis_maxC_prob']
    fixed_init_prob = gen_fixed_prob(learning_model._possible_configs, max_prob_config, prob_of_that_config)

    best_valid_error, test_error_at_best_epoch, best_epoch, epoch_bias = np.inf, np.inf, -1, 0
    train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = [], [], [], []
    config_curr_posterior_hist, config_prior_hist = [], []

    start_time = timeit.default_timer()
    for train_task_cnt, (task_for_train) in enumerate(task_training_order):
        print("\nTask - %d"%(task_for_train))
        tf.reset_default_graph()
        with tf.Session(config=config) as sess:
            learning_model.add_new_task(y_depth[task_for_train], task_for_train)
            task_model_index = learning_model.find_task_model(task_for_train)
            num_learned_tasks = learning_model.number_of_learned_tasks()

            sess.run(tf.global_variables_initializer())
            if save_graph:
                tfboard_writer = tf.summary.FileWriter('./graphs/%s/run%d/task%d'%(model_architecture, run_cnt, train_task_cnt), sess.graph)

            if not train_hyperpara['reset_prior']:
                accumulated_posterior, den_acc_posterior = np.zeros(len(learning_model._possible_configs), dtype=np.float32), 0.0

            while learning_step < min(learning_step_max, epoch_bias + patience):
                learning_step = learning_step+1

                #### training
                best_config = 0
                if learning_step > 0:
                    shuffle(indices[task_for_train])

                    for batch_cnt in range(num_train[task_for_train]//batch_size):
                        batch_train_x = train_data[task_for_train][0][indices[task_for_train][batch_cnt*batch_size:(batch_cnt+1)*batch_size], :]
                        batch_train_y = train_data[task_for_train][1][indices[task_for_train][batch_cnt*batch_size:(batch_cnt+1)*batch_size]]

                        if learning_model.task_is_new:
                            ## run update methods simultaneously (by adding grads)
                            sess.run(learning_model.update, feed_dict={learning_model.model_input[task_model_index]: batch_train_x, learning_model.true_output[task_model_index]: batch_train_y, learning_model.epoch: learning_step-1, learning_model.dropout_prob: 0.5, learning_model.posterior_placeholder: fixed_init_prob})
                        else:
                            sess.run(learning_model.update, feed_dict={learning_model.model_input[task_model_index]: batch_train_x, learning_model.true_output[task_model_index]: batch_train_y, learning_model.epoch: learning_step-1, learning_model.dropout_prob: 0.5})
                    if learning_model.task_is_new:
                        sess.run(learning_model.update_prior, feed_dict={learning_model.posterior_placeholder: fixed_init_prob})
                        best_config = learning_model.best_config(sess)
                    config_prior_hist.append(fixed_init_prob)
                    config_curr_posterior_hist.append(fixed_init_prob)

                #### performance evaluation
                train_error_tmp = [0.0 for _ in range(num_task)]
                validation_error_tmp = [0.0 for _ in range(num_task)]
                test_error_tmp = [0.0 for _ in range(num_task)]
                for tmp_cnt, (task_index_to_eval) in enumerate(task_training_order[:train_task_cnt+1]):
                    if task_index_to_eval in task_training_order[:tmp_cnt]:
                        continue
                    train_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, train_data[task_index_to_eval][0], train_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)
                    validation_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, validation_data[task_index_to_eval][0], validation_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)
                    test_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, test_data[task_index_to_eval][0], test_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)

                ## for classification, error_tmp is actually ACCURACY, thus, change the sign for checking improvement
                train_error, valid_error, test_error = -(sum(train_error_tmp)/(num_learned_tasks)), -(sum(validation_error_tmp)/(num_learned_tasks)), -(sum(test_error_tmp)/(num_learned_tasks))
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

                train_error_hist.append(train_error_tmp + [abs(train_error)])
                valid_error_hist.append(validation_error_tmp + [abs(valid_error)])
                test_error_hist.append(test_error_tmp + [abs(test_error)])
                best_test_error_hist.append(abs(test_error_at_best_epoch))

                #if learning_step >= epoch_bias+min(patience, learning_step_max//num_task):
                if learning_step >= epoch_bias+min(patience, learning_step_max//len(task_training_order)):
                    learning_model.convert_tfVar_to_npVar(sess)

                    if save_param:
                        para_file_name = param_folder_path + '/model_parameter_taskC%d_task%d.mat'%(train_task_cnt, task_for_train)
                        curr_param = learning_model.get_params_val(sess, use_npparams=True)
                        savemat(para_file_name, {'parameter': curr_param})

                    if train_task_cnt == len(task_training_order)-1:
                        if save_param:
                            para_file_name = param_folder_path + '/final_model_parameter.mat'
                            curr_param = learning_model.get_params_val(sess, use_npparams=True)
                            savemat(para_file_name, {'parameter': curr_param})
                    else:
                        # update epoch_bias, task_for_train, task_change_epoch
                        epoch_bias = learning_step
                        task_change_epoch.append(learning_step+1)

                        # initialize best_valid_error, best_epoch, patience
                        patience = train_hyperpara['patience']
                        best_valid_error, best_epoch = np.inf, -1

                        print('\n\t>>Change to new task!<<\n')
                        if learning_model.task_is_new:
                            print('\t', learning_model._possible_configs[best_config])
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
    result_summary['history_net_prior_in_epoch'] = config_curr_posterior_hist
    result_summary['history_net_prior'] = config_prior_hist
    result_summary['conv_sharing'] = learning_model.conv_sharing

    tmp_valid_error_hist = np.array(valid_error_hist)
    chk_epoch = [(task_change_epoch[x], task_change_epoch[x+1]) for x in range(len(task_change_epoch)-1)] + [(task_change_epoch[-1], learning_step+1)]
    #tmp_best_valid_error_list = [np.amax(tmp_valid_error_hist[x[0]:x[1], t]) for x, t in zip(chk_epoch, range(num_task))]
    #result_summary['best_validation_error'] = sum(tmp_best_valid_error_list) / float(len(tmp_best_valid_error_list))
    result_summary['task_changed_epoch'] = task_change_epoch

    if save_graph:
        tfboard_writer.close()

    return result_summary, learning_model.num_trainable_var




###################################################################################################################
###################################################################################################################
######## Learn which layers to share based on Neural Architecture Search (especially DARTS)
########     https://arxiv.org/abs/1806.09055
########     https://github.com/quark0/darts
###################################################################################################################
###################################################################################################################
def train_lifelong_DARTS_hybridNN(model_architecture, model_hyperpara, train_hyperpara, dataset, data_type, classification_prob, useGPU=False, GPU_device=0, save_param=False, param_folder_path='saved_param', save_param_interval=100, save_graph=False, tfInitParam=None, run_cnt=0):
    print("Training function for lifelong learning/Hybrid model with automatic sharing(DARTS)!")
    assert (model_architecture == 'darts_hps_cnn' or model_architecture == 'darts_dfcnn'), "Use train function appropriate to the architecture"

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
        raise ValueError


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
        return (None, None, None, None)

    ### Training Procedure
    best_param = []
    if save_param:
        best_para_file_name = param_folder_path+'/best_model_parameter'
        print("Saving trained parameters at '%s'" %(param_folder_path) )
    else:
        print("Not saving trained parameters")

    learning_step = -1
    if num_task > 1:
        indices = [list(range(num_train[x])) for x in range(num_task)]
    else:
        indices = [list(range(num_train[0]))]

    best_valid_error, test_error_at_best_epoch, best_epoch, epoch_bias = np.inf, np.inf, -1, 0
    train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = [], [], [], []
    selection_variable_hist = []

    start_time = timeit.default_timer()
    for train_task_cnt, (task_for_train) in enumerate(task_training_order):
        print("\nTask - %d"%(task_for_train))
        tf.reset_default_graph()
        with tf.Session(config=config) as sess:
            learning_model.add_new_task(y_depth[task_for_train], task_for_train)
            num_learned_tasks = learning_model.number_of_learned_tasks()

            sess.run(tf.global_variables_initializer())
            if save_graph:
                tfboard_writer = tf.summary.FileWriter('./graphs/%s/run%d/task%d'%(model_architecture, run_cnt, train_task_cnt), sess.graph)

            while learning_step < min(learning_step_max, epoch_bias + patience):
                learning_step = learning_step+1

                #### training
                if learning_step > 0:
                    learning_model.train_one_epoch(sess, train_data[task_for_train][0], train_data[task_for_train][1], learning_step-1, task_for_train, indices[task_for_train], dropout_prob=0.5)

                    if learning_model.task_is_new:
                        best_config = learning_model.best_config(sess)
                        selection_param_value = learning_model.get_darts_selection_val(sess)
                        selection_variable_hist.append(np.concatenate(selection_param_value))
                    else:
                        selection_variable_hist.append(np.zeros(2*learning_model.num_conv_layers, dtype=np.float32))

                #### performance evaluation
                train_error_tmp = [0.0 for _ in range(num_task)]
                validation_error_tmp = [0.0 for _ in range(num_task)]
                test_error_tmp = [0.0 for _ in range(num_task)]
                for tmp_cnt, (task_index_to_eval) in enumerate(task_training_order[:train_task_cnt+1]):
                    if task_index_to_eval in task_training_order[:tmp_cnt]:
                        continue
                    train_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, train_data[task_index_to_eval][0], train_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)
                    validation_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, validation_data[task_index_to_eval][0], validation_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)
                    test_error_tmp[task_index_to_eval] = learning_model.compute_accuracy_one_task(sess, test_data[task_index_to_eval][0], test_data[task_index_to_eval][1], task_index_to_eval, dropout_prob=1.0)

                ## for classification, error_tmp is actually ACCURACY, thus, change the sign for checking improvement
                train_error, valid_error, test_error = -(sum(train_error_tmp)/(num_learned_tasks)), -(sum(validation_error_tmp)/(num_learned_tasks)), -(sum(test_error_tmp)/(num_learned_tasks))
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

                train_error_hist.append(train_error_tmp + [abs(train_error)])
                valid_error_hist.append(validation_error_tmp + [abs(valid_error)])
                test_error_hist.append(test_error_tmp + [abs(test_error)])
                best_test_error_hist.append(abs(test_error_at_best_epoch))

                #if learning_step >= epoch_bias+min(patience, learning_step_max//num_task):
                if learning_step >= epoch_bias+min(patience, learning_step_max//len(task_training_order)):
                    learning_model.convert_tfVar_to_npVar(sess)

                    if save_param:
                        para_file_name = param_folder_path + '/model_parameter_taskC%d_task%d.mat'%(train_task_cnt, task_for_train)
                        curr_param = learning_model.get_params_val(sess, use_npparams=True)
                        savemat(para_file_name, {'parameter': curr_param})

                    if train_task_cnt == len(task_training_order)-1:
                        ## After training phase, compute mutual information
                        if save_param:
                            para_file_name = param_folder_path + '/final_model_parameter.mat'
                            curr_param = learning_model.get_params_val(sess, use_npparams=True)
                            savemat(para_file_name, {'parameter': curr_param})
                    else:
                        # update epoch_bias, task_for_train, task_change_epoch
                        epoch_bias = learning_step
                        task_change_epoch.append(learning_step+1)

                        # initialize best_valid_error, best_epoch, patience
                        patience = train_hyperpara['patience']
                        best_valid_error, best_epoch = np.inf, -1

                        print('\n\t>>Change to new task!<<\n')
                        if learning_model.task_is_new:
                            print('\t', learning_model._possible_configs[best_config])
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
    result_summary['history_DARTS_arch_select'] = selection_variable_hist
    result_summary['conv_sharing'] = learning_model.conv_sharing

    tmp_valid_error_hist = np.array(valid_error_hist)
    chk_epoch = [(task_change_epoch[x], task_change_epoch[x+1]) for x in range(len(task_change_epoch)-1)] + [(task_change_epoch[-1], learning_step+1)]
    #tmp_best_valid_error_list = [np.amax(tmp_valid_error_hist[x[0]:x[1], t]) for x, t in zip(chk_epoch, range(num_task))]
    #result_summary['best_validation_error'] = sum(tmp_best_valid_error_list) / float(len(tmp_best_valid_error_list))
    result_summary['task_changed_epoch'] = task_change_epoch

    if save_graph:
        tfboard_writer.close()

    return result_summary, learning_model.num_trainable_var