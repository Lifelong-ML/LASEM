import numpy as np
import tensorflow as tf
import sys
from psutil import Process
from os import getpid
from math import sqrt

_tf_ver = tf.__version__.split('.')
if int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) > 14):
    from tensorflow.compat.v1 import trainable_variables
    _tf_tensor = tf.is_tensor
else:
    from tensorflow import trainable_variables
    _tf_tensor = tf.contrib.framework.is_tensor

def get_memory_usage():
    return Process(getpid()).memory_percent()

def convert_array_to_oneHot(arr, n_classes):
    n_data = arr.shape[0]
    one_hot_arr = np.zeros((n_data, n_classes), dtype=np.float32)
    one_hot_arr[np.arange(n_data), np.round(arr).astype(np.int32)] = 1.0
    return one_hot_arr

def convert_dataset_to_oneHot(dataset, n_classes):
    num_tasks = len(dataset)
    if sys.version_info.major < 3:
        from types import ListType
        if type(n_classes) is not ListType:
            n_classes = [n_classes for _ in range(num_tasks)]
        else:
            assert (num_tasks == len(n_classes)), "Given n_classes doesn't match to the number of tasks"
    else:
        if type(n_classes) is not list:
            n_classes = [n_classes for _ in range(num_tasks)]
        else:
            assert (num_tasks == len(n_classes)), "Given n_classes doesn't match to the number of tasks"

    new_dataset = []
    for task_cnt in range(num_tasks):
        orig_x, orig_y = dataset[task_cnt]
        new_y = convert_array_to_oneHot(orig_y, n_classes[task_cnt])
        new_dataset.append( (orig_x, new_y) )
    return new_dataset


def l1_loss(x):
    n = 1
    for e in x.get_shape():
        n = n*int(e)
    return tf.reduce_sum(tf.abs(x))/n

#### function only for APD model (pruning mask/parameter whose absolute (L1) value is smaller than criterion)
def l1_pruning(x, crit):
    threshold = tf.cast(tf.greater(tf.abs(x), crit), tf.float32)
    return tf.multiply(x, threshold)


##########################################################
#### function to compute entropy from logit (softmax) ####
####           code from OpenAI's baselines           ####
##########################################################
def cat_softmax_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

##########################################################
#### function to compute entropy from logit (sigmoid) ####
##########################################################
def cat_sigmoid_entropy(logits):
    z0 = 1.0+tf.exp(-logits)
    p0 = 1.0/z0
    elementwise_entropy = (1.0-p0)*logits + tf.log(z0)
    return tf.reduce_sum(elementwise_entropy)


##############################################
#####  functions to save NN's parameter  #####
##############################################
def get_list_of_valid_tensors(list_of_variables):
    list_of_valid_tensors = []
    for elem in list_of_variables:
        #if elem is not None:
        if elem in tf.global_variables():
            list_of_valid_tensors.append(elem)
    return list_of_valid_tensors

def get_value_of_valid_tensors(tf_sess, list_of_variables):
    list_of_val = []
    for elem in list_of_variables:
        list_of_val.append(elem if (elem is None) else tf_sess.run(elem))
    return list_of_val

def savemat_wrapper(list_of_data):
    data_to_save = np.zeros((len(list_of_data),), dtype=np.object)
    for cnt in range(len(list_of_data)):
        if list_of_data[cnt] is not None:
            data_to_save[cnt] = list_of_data[cnt]
    return data_to_save

def savemat_wrapper_nested_list(list_of_data):
    data_to_save = np.zeros((len(list_of_data),), dtype=np.object)
    for cnt in range(len(list_of_data)):
        data_to_save[cnt] = savemat_wrapper(list_of_data[cnt])
    return data_to_save


##############################################
#####  functions to load NN's parameter  #####
##############################################
def reformat_list_params_for_loading(loaded_param_struct, model_architecture, model_hyperpara, num_tasks, for_visualize_saliency=False):
    num_fc_layers = len(model_hyperpara['hidden_layer']) + 1 # output layer hasn't been included
    temp = [len(model_hyperpara['kernel_sizes'])//2, len(model_hyperpara['stride_sizes'])//2, len(model_hyperpara['channel_sizes']), len(model_hyperpara['pooling_size'])//2]
    temp_cond = [(temp[i]==temp[i+1]) for i in range(len(temp)-1)]
    assert (all(temp_cond)), "Given hyper-parameters of conv layers mismatch each other!"
    num_conv_layers = temp[0]

    param_struct = {}
    if for_visualize_saliency:
        if model_architecture == 'mtl_several_cnn_minibatch':
            # 'conv_trainable_weights', 'fc_weights'
            temp_conv_params, temp_fc_params = loaded_param_struct['conv_trainable_weights'][0][0][0], loaded_param_struct['fc_weights'][0][0][0]
            assert (len(temp_conv_params)==2*num_conv_layers*num_tasks), "Given conv parameters are more/less than required!"
            assert (len(temp_fc_params)==2*num_fc_layers*num_tasks), "Given fc parameters are more/less than required!"

            param_struct['conv_params'] = [temp_conv_params[2*num_conv_layers*t:2*num_conv_layers*(t+1)] for t in range(num_tasks)]
            param_struct['fc_params'] = [temp_fc_params[2*num_fc_layers*t:2*num_fc_layers*(t+1)] for t in range(num_tasks)]
        elif model_architecture == 'mtl_cnn_hps_minibatch':
            # 'conv_trainable_weights', 'fc_weights'
            temp_conv_params, temp_fc_params = loaded_param_struct['conv_trainable_weights'][0][0][0], loaded_param_struct['fc_weights'][0][0][0]
            param_struct['conv_params'] = [temp_conv_params for _ in range(num_tasks)]
            param_struct['fc_params'] = [temp_fc_params[t][0] for t in range(num_tasks)]
        elif model_architecture == 'll_cnn_progressive_minibatch':
            print("\nNot yet implemented!\n")
        elif model_architecture == 'll_cnn_dynamically_expandable_minibatch':
            print("\nNot yet implemented!\n")
        elif model_architecture == 'll_cnn_elastic_weight_consolidation_minibatch':
            print("\nNot yet implemented!\n")
        elif 'deconv' in model_architecture:
            if ('flexible' in model_architecture) and (model_hyperpara['highway_connect']>0):
                # 'conv_KB', 'conv_TS', 'conv_trainable_weights', 'conv_highway_weights', 'fc_weights'
                print("\nNot yet implemented!\n")
            elif 'flexible' in model_architecture:
                # 'conv_KB', 'conv_TS', 'conv_trainable_weights', 'fc_weights'
                temp_conv_params, temp_fc_params = loaded_param_struct['conv_trainable_weights'][0][0][0], loaded_param_struct['fc_weights'][0][0][0]
                temp_gen_conv_params = loaded_param_struct['conv_generated_weights'][0][0][0]
                temp_conv_cond = [len(temp_conv_params[i][0])==2*num_conv_layers for i in range(len(temp_conv_params))]
                temp_gen_conv_cond = [len(temp_gen_conv_params[i][0])==2*num_conv_layers for i in range(len(temp_gen_conv_params))]
                temp_fc_cond = [len(temp_fc_params[i][0])==2*num_fc_layers for i in range(len(temp_fc_params))]
                assert (len(temp_conv_params)==num_tasks and all(temp_conv_cond) and len(temp_gen_conv_params)==num_tasks and all(temp_gen_conv_cond)), "Given conv parameters are more/less than required!"
                assert (len(temp_fc_params)==num_tasks and all(temp_fc_cond)), "Given fc parameters are more/less than required!"

                conv_sharing = model_hyperpara['conv_sharing']
                param_struct['conv_params'] = [[temp_gen_conv_params[t][0][l] if conv_sharing[l//2] else temp_conv_params[t][0][l] for l in range(2*num_conv_layers)] for t in range(num_tasks)]
                param_struct['fc_params'] = [temp_fc_params[t][0] for t in range(num_tasks)]
            else:
                # 'conv_KB', 'conv_TS', 'fc_weights'
                print("\nNot yet implemented!\n")

        model_architecture = 'mtl_several_cnn_minibatch'
    else:
        print("\nNot yet implemented!\n")
    return model_architecture, param_struct


############################################
#####     functions for saving data    #####
############################################
def tflized_data(data_list, do_MTL, num_tasks=0):
    with tf.name_scope('RawData_Input'):
        if not do_MTL:
            #### single task
            assert (len(data_list) == 6), "Data given to model isn't in the right format"
            train_x = tf.constant(data_list[0], dtype=tf.float32)
            train_y = tf.constant(data_list[1], dtype=tf.float32)
            valid_x = tf.constant(data_list[2], dtype=tf.float32)
            valid_y = tf.constant(data_list[3], dtype=tf.float32)
            test_x = tf.constant(data_list[4], dtype=tf.float32)
            test_y = tf.constant(data_list[5], dtype=tf.float32)
        else:
            #### multi-task
            if num_tasks < 2:
                train_x = [tf.constant(data_list[0], dtype=tf.float32)]
                train_y = [tf.constant(data_list[1], dtype=tf.float32)]
                valid_x = [tf.constant(data_list[2], dtype=tf.float32)]
                valid_y = [tf.constant(data_list[3], dtype=tf.float32)]
                test_x = [tf.constant(data_list[4], dtype=tf.float32)]
                test_y = [tf.constant(data_list[5], dtype=tf.float32)]
            else:
                train_x = [tf.constant(data_list[0][x][0], dtype=tf.float32) for x in range(num_tasks)]
                train_y = [tf.constant(data_list[0][x][1], dtype=tf.float32) for x in range(num_tasks)]
                valid_x = [tf.constant(data_list[1][x][0], dtype=tf.float32) for x in range(num_tasks)]
                valid_y = [tf.constant(data_list[1][x][1], dtype=tf.float32) for x in range(num_tasks)]
                test_x = [tf.constant(data_list[2][x][0], dtype=tf.float32) for x in range(num_tasks)]
                test_y = [tf.constant(data_list[2][x][1], dtype=tf.float32) for x in range(num_tasks)]
    return [train_x, train_y, valid_x, valid_y, test_x, test_y]


def minibatched_data(data_list, batch_size, data_index, do_MTL, num_tasks=0):
    with tf.name_scope('Minibatch_Data'):
        if not do_MTL:
            #### single task
            train_x_batch = tf.slice(data_list[0], [batch_size * data_index, 0], [batch_size, -1])
            valid_x_batch = tf.slice(data_list[2], [batch_size * data_index, 0], [batch_size, -1])
            test_x_batch = tf.slice(data_list[4], [batch_size * data_index, 0], [batch_size, -1])

            if len(data_list[1].shape) == 1:
                train_y_batch = tf.slice(data_list[1], [batch_size*data_index], [batch_size])
                valid_y_batch = tf.slice(data_list[3], [batch_size*data_index], [batch_size])
                test_y_batch = tf.slice(data_list[5], [batch_size*data_index], [batch_size])
            else:
                train_y_batch = tf.slice(data_list[1], [batch_size*data_index, 0], [batch_size, -1])
                valid_y_batch = tf.slice(data_list[3], [batch_size*data_index, 0], [batch_size, -1])
                test_y_batch = tf.slice(data_list[5], [batch_size*data_index, 0], [batch_size, -1])
        else:
            #### multi-task
            train_x_batch = [tf.slice(data_list[0][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
            valid_x_batch = [tf.slice(data_list[2][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
            test_x_batch = [tf.slice(data_list[4][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]

            if len(data_list[1][0].shape) == 1:
                train_y_batch = [tf.slice(data_list[1][x], [batch_size*data_index], [batch_size]) for x in range(num_tasks)]
                valid_y_batch = [tf.slice(data_list[3][x], [batch_size*data_index], [batch_size]) for x in range(num_tasks)]
                test_y_batch = [tf.slice(data_list[5][x], [batch_size*data_index], [batch_size]) for x in range(num_tasks)]
            else:
                train_y_batch = [tf.slice(data_list[1][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
                valid_y_batch = [tf.slice(data_list[3][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
                test_y_batch = [tf.slice(data_list[5][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
    return (train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch)


def minibatched_data_not_in_gpu(data_x, data_y, do_MTL):
    with tf.name_scope('Minibatch_Data_CNN'):
        if not do_MTL:
            #### single task
            train_x_batch, valid_x_batch, test_x_batch = data_x, data_x, data_x

            train_y_batch, valid_y_batch, test_y_batch = data_y, data_y, data_y
        else:
            #### multi-task
            train_x_batch = [x for x in data_x]
            valid_x_batch = [x for x in data_x]
            test_x_batch = [x for x in data_x]

            train_y_batch = [y for y in data_y]
            valid_y_batch = [y for y in data_y]
            test_y_batch = [y for y in data_y]
    return (train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch)


def minibatched_cnn_data(data_list, batch_size, data_index, data_tensor_dim, do_MTL, num_tasks=0):
    with tf.name_scope('Minibatch_Data_CNN'):
        if not do_MTL:
            #### single task
            train_x_batch = tf.reshape(tf.slice(data_list[0], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim)
            valid_x_batch = tf.reshape(tf.slice(data_list[2], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim)
            test_x_batch = tf.reshape(tf.slice(data_list[4], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim)

            if len(data_list[1].shape) == 1:
                train_y_batch = tf.slice(data_list[1], [batch_size*data_index], [batch_size])
                valid_y_batch = tf.slice(data_list[3], [batch_size*data_index], [batch_size])
                test_y_batch = tf.slice(data_list[5], [batch_size*data_index], [batch_size])
            else:
                train_y_batch = tf.slice(data_list[1], [batch_size*data_index, 0], [batch_size, -1])
                valid_y_batch = tf.slice(data_list[3], [batch_size*data_index, 0], [batch_size, -1])
                test_y_batch = tf.slice(data_list[5], [batch_size*data_index, 0], [batch_size, -1])
        else:
            #### multi-task
            train_x_batch = [tf.reshape(tf.slice(data_list[0][x], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim) for x in range(num_tasks)]
            valid_x_batch = [tf.reshape(tf.slice(data_list[2][x], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim) for x in range(num_tasks)]
            test_x_batch = [tf.reshape(tf.slice(data_list[4][x], [batch_size*data_index, 0], [batch_size, -1]), data_tensor_dim) for x in range(num_tasks)]

            if len(data_list[1][0].shape) == 1:
                train_y_batch = [tf.slice(data_list[1][x], [batch_size*data_index], [batch_size]) for x in range(num_tasks)]
                valid_y_batch = [tf.slice(data_list[3][x], [batch_size*data_index], [batch_size]) for x in range(num_tasks)]
                test_y_batch = [tf.slice(data_list[5][x], [batch_size*data_index], [batch_size]) for x in range(num_tasks)]
            else:
                train_y_batch = [tf.slice(data_list[1][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
                valid_y_batch = [tf.slice(data_list[3][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
                test_y_batch = [tf.slice(data_list[5][x], [batch_size*data_index, 0], [batch_size, -1]) for x in range(num_tasks)]
    return (train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch)


def minibatched_cnn_data_not_in_gpu(data_x, data_y, data_tensor_dim, do_MTL, num_tasks=0):
    with tf.name_scope('Minibatch_Data_CNN'):
        if not do_MTL:
            #### single task
            train_x_batch = tf.reshape(data_x, data_tensor_dim)
            valid_x_batch = tf.reshape(data_x, data_tensor_dim)
            test_x_batch = tf.reshape(data_x, data_tensor_dim)

            train_y_batch = data_y
            valid_y_batch = data_y
            test_y_batch = data_y
        else:
            #### multi-task
            train_x_batch = [tf.reshape(x, data_tensor_dim) for x in data_x]
            valid_x_batch = [tf.reshape(x, data_tensor_dim) for x in data_x]
            test_x_batch = [tf.reshape(x, data_tensor_dim) for x in data_x]
            #[tf.reshape(data_x, data_tensor_dim) for _ in range(num_tasks)]

            train_y_batch = [y for y in data_y]
            valid_y_batch = [y for y in data_y]
            test_y_batch = [y for y in data_y]
    return (train_x_batch, train_y_batch, valid_x_batch, valid_y_batch, test_x_batch, test_y_batch)



############################################
#### functions for (MTL) model's output ####
############################################
def mtl_model_output_functions(models, y_batches, num_tasks, dim_output, classification=False):
    if classification:
        with tf.name_scope('Model_Eval'):
            train_eval = [tf.nn.softmax(models[0][x][-1]) for x in range(num_tasks)]
            valid_eval = [tf.nn.softmax(models[1][x][-1]) for x in range(num_tasks)]
            test_eval = [tf.nn.softmax(models[2][x][-1]) for x in range(num_tasks)]

            train_output_label = [tf.argmax(models[0][x][-1], 1) for x in range(num_tasks)]
            valid_output_label = [tf.argmax(models[1][x][-1], 1) for x in range(num_tasks)]
            test_output_label = [tf.argmax(models[2][x][-1], 1) for x in range(num_tasks)]

        with tf.name_scope('Model_Loss'):
            train_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batches[0][x], tf.int32), logits=models[0][x][-1]) for x in range(num_tasks)]
            valid_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batches[1][x], tf.int32), logits=models[1][x][-1]) for x in range(num_tasks)]
            test_loss = [tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batches[2][x], tf.int32), logits=models[2][x][-1]) for x in range(num_tasks)]

        with tf.name_scope('Model_Accuracy'):
            train_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(models[0][x][-1], 1), tf.cast(y_batches[0][x], tf.int64)), tf.float32)) for x in range(num_tasks)]
            valid_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(models[1][x][-1], 1), tf.cast(y_batches[1][x], tf.int64)), tf.float32)) for x in range(num_tasks)]
            test_accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(models[2][x][-1], 1), tf.cast(y_batches[2][x], tf.int64)), tf.float32)) for x in range(num_tasks)]
    else:
        with tf.name_scope('Model_Eval'):
            train_eval = [models[0][x][-1] for x in range(num_tasks)]
            valid_eval = [models[1][x][-1] for x in range(num_tasks)]
            test_eval = [models[2][x][-1] for x in range(num_tasks)]

        with tf.name_scope('Model_Loss'):
            train_loss = [2.0* tf.nn.l2_loss(train_eval[x]-y_batches[0][x]) for x in range(num_tasks)]
            valid_loss = [2.0* tf.nn.l2_loss(valid_eval[x]-y_batches[1][x]) for x in range(num_tasks)]
            test_loss = [2.0* tf.nn.l2_loss(test_eval[x]-y_batches[2][x]) for x in range(num_tasks)]

        train_accuracy, valid_accuracy, test_accuracy = None, None, None
        train_output_label, valid_output_label, test_output_label = None, None, None
    return (train_eval, valid_eval, test_eval, train_loss, valid_loss, test_loss, train_accuracy, valid_accuracy, test_accuracy, train_output_label, valid_output_label, test_output_label)




############################################
####  functions for Data Augmentation   ####
############################################
def data_augmentation(data_x, data_y, data_amount_to_add, image_dimension, sample_with_repetition=False):
    num_orig_data, x_dim = data_x.shape
    new_x, new_y = [], []
    for cnt in range(data_amount_to_add):
        data_index = np.random.randint(0, num_orig_data, dtype=np.int64) if sample_with_repetition else cnt
        augment_type = np.random.randint(0, 4, dtype=np.int64)
        x, y = data_x[data_index, :], data_y[data_index]
        if augment_type < 1:
            new_x.append(np.array(x))
        elif augment_type < 2:
            ## add small noise
            new_x.append(x+np.random.randn(x_dim)*0.1)
        elif augment_type < 3:
            ## flip
            tmp_x, new_x_tmp = np.reshape(x, image_dimension), np.zeros(image_dimension)
            for row_cnt in range(image_dimension[0]):
                for col_cnt in range(image_dimension[1]):
                    new_x_tmp[image_dimension[0]-1-row_cnt, image_dimension[1]-1-col_cnt, :] = tmp_x[row_cnt, col_cnt, :]
            new_x.append(np.reshape(new_x_tmp, [x_dim]))
        elif augment_type < 4:
            ## zero padding and crop
            flip_img_or_not, pad_size = np.random.rand(), 4
            padded_dimension = [image_dimension[0]+pad_size*2, image_dimension[1]+pad_size*2, image_dimension[2]]
            tmp_x, new_x_tmp = np.reshape(x, image_dimension), np.zeros(padded_dimension)
            if flip_img_or_not < 0.5:
                new_x_tmp[pad_size:pad_size+image_dimension[0], pad_size:pad_size+image_dimension[1], :] = tmp_x
            else:
                for row_cnt in range(image_dimension[0]):
                    for col_cnt in range(image_dimension[1]):
                        new_x_tmp[pad_size+image_dimension[0]-1-row_cnt, pad_size+image_dimension[1]-1-col_cnt, :] = tmp_x[row_cnt, col_cnt, :]
            crop_row_ind, crop_col_ind = np.random.randint(0, 2*pad_size, dtype=np.int64), np.random.randint(0, 2*pad_size, dtype=np.int64)
            new_x.append(np.reshape(new_x_tmp[crop_row_ind:crop_row_ind+image_dimension[0], crop_col_ind:crop_col_ind+image_dimension[1], :], [x_dim]))
        new_y.append(y)
    return np.array(new_x, dtype=np.float32), np.array(new_y, dtype=np.float32)


def data_augmentation_in_minibatch(data_x, data_y, image_dimension):
    num_orig_data, x_dim = data_x.shape
    new_x, new_y = data_augmentation(data_x, data_y, num_orig_data, image_dimension, sample_with_repetition=False)
    return np.array(new_x, dtype=np.float32), np.array(new_y, dtype=np.float32)

def data_augmentation_STL_analysis(dataset, task_to_augment, data_ratio_after_add, image_dimension):
    train_data, validation_data, test_data = dataset
    orig_train_x, orig_train_y = train_data[task_to_augment][0], train_data[task_to_augment][1]

    num_total_train_data_upto_T = sum([train_data[x][0].shape[0] for x in range(task_to_augment+1)])
    num_total_train_data = int(num_total_train_data_upto_T*(float(data_ratio_after_add)))
    if num_total_train_data > train_data[task_to_augment][0].shape[0]:
        num_new_train_data = num_total_train_data - train_data[task_to_augment][0].shape[0]
        new_train_x_tmp, new_train_y_tmp = data_augmentation(orig_train_x, orig_train_y, num_new_train_data, image_dimension, sample_with_repetition=True)
        new_train_x = np.concatenate((orig_train_x, new_train_x_tmp), axis=0)
        new_train_y = np.concatenate((orig_train_y, new_train_y_tmp), axis=0)
        train_data[task_to_augment] = (new_train_x, new_train_y)
    return (train_data, validation_data, test_data)


########################################################
####  functions to fill zeros to match batch size   ####
########################################################
def data_x_add_dummy(data_x, batch_size):
    data_shape = data_x.shape

    output_shape = list(data_shape)
    output_shape[0] = batch_size
    data_with_dummy = np.zeros(output_shape, dtype=data_x.dtype)
    data_with_dummy[0:data_shape[0]] = data_x
    return data_with_dummy

def data_x_and_y_add_dummy(data_x, data_y, batch_size):
    x_shape, y_shape = data_x.shape, data_y.shape
    out_x_shape, out_y_shape = list(x_shape), list(y_shape)
    out_x_shape[0], out_y_shape[0] = batch_size, batch_size

    x_with_dummy, y_with_dummy = np.zeros(out_x_shape, dtype=data_x.dtype), np.zeros(out_y_shape, dtype=data_y.dtype)
    x_with_dummy[0:x_shape[0]], y_with_dummy[0:y_shape[0]] = data_x, data_y
    return x_with_dummy, y_with_dummy




############################################
####   Basic functions for Neural Net   ####
############################################
_he_init = False
#_weight_init_stddev = 0.001
_weight_init_stddev = 0.05
_bias_init_val = 0.2
#_bias_init_val = 0.1

#### Leaky ReLu function
def leaky_relu(function_in, leaky_alpha=0.01):
    return tf.nn.relu(function_in) - leaky_alpha*tf.nn.relu(-function_in)

#### function to compute entropy
def cat_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

#### function to generate weight parameter
def new_placeholder(shape):
    return tf.placeholder(shape=shape, dtype=tf.float32)

#### function to generate weight parameter
def new_weight(shape, trainable=True, init_tensor=None, name=None):
    if init_tensor is None:
        if _he_init:
            n_in, n_out = shape[-2], shape[-1]
            scale = sqrt(2.0/float(n_in))
            #scale = sqrt(2.0/float(n_in+n_out)) if len(shape) == 4 else 0.01*sqrt(2.0/float(n_in+n_out))
            return tf.Variable(tf.random.uniform(shape, maxval=scale), trainable=trainable, name=name)
        else:
            return tf.Variable(tf.truncated_normal(shape, stddev=_weight_init_stddev) if init_tensor is None else init_tensor, trainable=trainable, name=name)
    else:
        return tf.Variable(init_tensor, trainable=trainable, name=name)

#### function to generate bias parameter
def new_bias(shape, trainable=True, init_val=_bias_init_val, init_tensor=None, name=None):
    return tf.Variable(tf.constant(init_val, dtype=tf.float32, shape=shape) if init_tensor is None else init_tensor, trainable=trainable, name=name)

#### function to generate knowledge-base parameters for ELLA_tensorfactor layer
def new_ELLA_KB_param(shape, layer_number, task_number, reg_type, init_tensor=None, trainable=True):
    #kb_name = 'KB_'+str(layer_number)+'_'+str(task_number)
    kb_name = 'KB_'+str(layer_number)
    if init_tensor is None:
        param_to_return = tf.get_variable(name=kb_name, shape=shape, dtype=tf.float32, regularizer=reg_type, trainable=trainable)
    elif type(init_tensor) == np.ndarray:
        param_to_return = tf.get_variable(name=kb_name, shape=shape, dtype=tf.float32, regularizer=reg_type, initializer=tf.constant_initializer(init_tensor), trainable=trainable)
    else:
        param_to_return = init_tensor
    return param_to_return

#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_ELLA_cnn_deconv_tensordot_TS_param(shape, layer_number, task_number, reg_type, init_tensor, trainable):
    ts_w_name, ts_b_name, ts_k_name, ts_p_name = 'TS_DeconvW0_'+str(layer_number)+'_'+str(task_number), 'TS_Deconvb0_'+str(layer_number)+'_'+str(task_number), 'TS_ConvW1_'+str(layer_number)+'_'+str(task_number), 'TS_Convb0_'+str(layer_number)+'_'+str(task_number)
    params_to_return, params_name = [], [ts_w_name, ts_b_name, ts_k_name, ts_p_name]
    for i, (t, n) in enumerate(zip(init_tensor, params_name)):
        if t is None:
            params_to_return.append(tf.get_variable(name=n, shape=shape[i], dtype=tf.float32, regularizer=reg_type if trainable and i<3 else None, trainable=trainable))
        elif type(t) == np.ndarray:
            params_to_return.append(tf.get_variable(name=n, shape=shape[i], dtype=tf.float32, regularizer=reg_type if trainable and i<3 else None, trainable=trainable, initializer=tf.constant_initializer(t)))
        else:
            params_to_return.append(t)
    return params_to_return


############################################################
#####   functions for adding fully-connected network   #####
############################################################
#### function to add fully-connected layer
def new_fc_layer(layer_input, output_dim, activation_fn=tf.nn.relu, weight=None, bias=None, trainable=True, use_numpy_var_in_graph=False):
    input_dim = int(layer_input.shape[1])
    with tf.name_scope('fc_layer'):
        if weight is None:
            weight = new_weight(shape=[input_dim, output_dim], trainable=trainable)
        elif (type(weight) == np.ndarray) and not use_numpy_var_in_graph:
            weight = new_weight(shape=[input_dim, output_dim], init_tensor=weight, trainable=trainable)
        if bias is None:
            bias = new_bias(shape=[output_dim], trainable=trainable)
        elif (type(bias) == np.ndarray) and not use_numpy_var_in_graph:
            bias = new_bias(shape=[output_dim], init_tensor=bias, trainable=trainable)

        if activation_fn is None:
            layer = tf.matmul(layer_input, weight) + bias
        elif activation_fn is 'classification':
            layer = tf.matmul(layer_input, weight) + bias
        else:
            layer = activation_fn( tf.matmul(layer_input, weight) + bias )
    return layer, [weight, bias]

#### function to generate network of fully-connected layers
####      'dim_layers' contains input/output layer
def new_fc_net(net_input, dim_layers, activation_fn=tf.nn.relu, params=None, output_type=None, tensorboard_name_scope='fc_net', trainable=True, use_numpy_var_in_graph=False):
    if params is None:
        params = [None for _ in range(2*len(dim_layers))]

    layers, params_to_return = [], []
    if len(dim_layers) < 1:
        #### for the case that hard-parameter shared network does not have shared layers
        layers.append(net_input)
    else:
        with tf.name_scope(tensorboard_name_scope):
            for cnt in range(len(dim_layers)):
                if cnt == 0:
                    layer_tmp, para_tmp = new_fc_layer(net_input, dim_layers[cnt], activation_fn=activation_fn, weight=params[0], bias=params[1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif cnt == len(dim_layers)-1 and output_type is 'classification':
                    layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt],  activation_fn='classification', weight=params[2*cnt], bias=params[2*cnt+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif cnt == len(dim_layers)-1 and output_type is None:
                    layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], activation_fn=None, weight=params[2*cnt], bias=params[2*cnt+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif cnt == len(dim_layers)-1 and output_type is 'same':
                    layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], activation_fn=activation_fn, weight=params[2*cnt], bias=params[2*cnt+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                else:
                    layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], activation_fn=activation_fn, weight=params[2*cnt], bias=params[2*cnt+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                layers.append(layer_tmp)
                params_to_return = params_to_return + para_tmp
    return (layers, params_to_return)

############################################################
#####    functions for adding convolutional network    #####
############################################################
#### function to add 2D convolutional layer
def new_cnn_layer(layer_input, k_size, stride_size=[1, 1, 1, 1], activation_fn=tf.nn.relu, weight=None, bias=None, padding_type='SAME', max_pooling=False, pool_size=None, skip_connect_input=None, highway_connect_type=0, highway_gate=None, trainable=True, use_numpy_var_in_graph=False, name_scope='conv_layer'):
    with tf.name_scope(name_scope):
        if weight is None:
            weight = new_weight(shape=k_size, trainable=trainable)
        elif (type(weight) == np.ndarray) and not use_numpy_var_in_graph:
            weight = new_weight(shape=k_size, init_tensor=weight, trainable=trainable)
        if bias is None:
            bias = new_bias(shape=[k_size[-1]], trainable=trainable)
        elif (type(bias) == np.ndarray) and not use_numpy_var_in_graph:
            bias = new_bias(shape=[k_size[-1]], init_tensor=bias, trainable=trainable)

        conv_layer = tf.nn.conv2d(layer_input, weight, strides=stride_size, padding=padding_type) + bias

        if not (activation_fn is None):
            conv_layer = activation_fn(conv_layer)

        if skip_connect_input is not None:
            shape1, shape2 = conv_layer.get_shape().as_list(), skip_connect_input.get_shape().as_list()
            assert (len(shape1) == len(shape2)), "Shape of layer's output and input of skip connection do not match!"
            assert (all([(x==y) for (x, y) in zip(shape1, shape2)])), "Shape of layer's output and input of skip connection do NOT match!"
            conv_layer = conv_layer + skip_connect_input

        if (highway_connect_type > 0) and (highway_gate is not None):
            conv_layer = tf.multiply(conv_layer, highway_gate) + tf.multiply(layer_input, 1-highway_gate)

        if max_pooling and (pool_size[1] > 1 or pool_size[2] > 1):
            layer = tf.nn.max_pool(conv_layer, ksize=pool_size, strides=pool_size, padding=padding_type)
        else:
            layer = conv_layer
    return (layer, [weight, bias])

#### function to generate network of convolutional layers
####      conv-pool-conv-pool-...-conv-pool-flat-dropout
####      k_sizes/stride_size/pool_sizes : [x_0, y_0, x_1, y_1, ..., x_m, y_m]
####      ch_sizes : [img_ch, ch_0, ch_1, ..., ch_m]
def new_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=tf.nn.relu, params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0], skip_connections=[], trainable=True, use_numpy_var_in_graph=False):
    if not max_pool:
        pool_sizes = [None for _ in range(len(k_sizes))]

    if params is None:
        params = [None for _ in range(len(k_sizes))]

    layers_for_skip, next_skip_connect = [net_input], None
    layers, params_to_return = [], []
    with tf.name_scope('conv_net'):
        if len(k_sizes) < 2:
            #### for the case that hard-parameter shared network does not have shared layers
            layers.append(net_input)
        else:
            for layer_cnt in range(len(k_sizes)//2):
                if next_skip_connect is None and len(skip_connections) > 0:
                    next_skip_connect = skip_connections.pop(0)
                if next_skip_connect is not None:
                    skip_connect_in, skip_connect_out = next_skip_connect
                    assert (skip_connect_in > -1 and skip_connect_out > -1), "Given skip connection has error (try connecting non-existing layer)"
                else:
                    skip_connect_in, skip_connect_out = -1, -1

                if layer_cnt == skip_connect_out:
                    processed_skip_connect_input = layers_for_skip[skip_connect_in]
                    for layer_cnt_tmp in range(skip_connect_in, skip_connect_out):
                        if max_pool and (pool_sizes[2*layer_cnt_tmp]>1 or pool_sizes[2*layer_cnt_tmp+1]>1):
                            processed_skip_connect_input = tf.nn.max_pool(processed_skip_connect_input, ksize=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], strides=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], padding=padding_type)
                else:
                    processed_skip_connect_input = None

                if layer_cnt == 0:
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, weight=params[2*layer_cnt], bias=params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                else:
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=layers[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, weight=params[2*layer_cnt], bias=params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                layers.append(layer_tmp)
                layers_for_skip.append(layer_tmp)
                params_to_return = params_to_return + para_tmp
                if layer_cnt == skip_connect_out:
                    next_skip_connect = None

        #### flattening output
        if flat_output:
            output_dim = [int(layers[-1].shape[1]*layers[-1].shape[2]*layers[-1].shape[3])]
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))
        else:
            output_dim = layers[-1].shape[1:]

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, params_to_return, output_dim)


#### function to count trainable parameters in computational graph
def count_trainable_var():
    total_para_cnt = 0
    for variable in trainable_variables():
        para_cnt_tmp = 1
        for dim in variable.get_shape():
            para_cnt_tmp = para_cnt_tmp * dim.value
        total_para_cnt = total_para_cnt + para_cnt_tmp
    return total_para_cnt

def count_trainable_var2(list_params):
    total_para_cnt = 0
    for var in list_params:
        para_cnt_tmp = 1
        if type(var) == np.ndarray:
            for dim in var.shape:
                para_cnt_tmp *= int(dim)
        elif _tf_tensor(var):
            for dim in var.get_shape():
                para_cnt_tmp *= int(dim)
        else:
            para_cnt_tmp = 0
        total_para_cnt += para_cnt_tmp
    return total_para_cnt


############################################
####  functions for Mutual Information  ####
############################################
def compute_kernel_matrix(data_x, h_val):
    n, d = data_x.shape
    sigma = h_val * ( (float(n))**(-1.0/(4.0+float(d))) )
    K_matrix = np.ones([n, n], dtype=np.float64)
    for row_cnt in range(n):
        for col_cnt in range(row_cnt+1, n):
            diff_norm = np.linalg.norm(data_x[row_cnt,:] - data_x[col_cnt,:])
            v = np.exp( -(diff_norm**2)/(2.0*sigma*sigma) )
            K_matrix[row_cnt, col_cnt] = K_matrix[col_cnt, row_cnt] = v
    return K_matrix

def compute_kernel_matrices(data_x_3d, h_val):
    ## return K1 \dot K2 \dot ... \dot Kc where \dot is Hadamard product
    c = data_x_3d.shape[-1]
    K = compute_kernel_matrix(data_x_3d[:,:,0], h_val)
    for channel_cnt in range(1, c):
        K *= compute_kernel_matrix(data_x_3d[:,:,channel_cnt], h_val)
    return K

def compute_information(alpha_val, h_val, data_x_3d=None, kernel_x=None):
    assert ((data_x_3d is not None) or (kernel_x is not None)), "Give either raw data or computed kernel matrix"
    if kernel_x is None:
        kernel_x = compute_kernel_matrices(data_x_3d, h_val)

    temp_K = kernel_x / np.trace(kernel_x)
    lambdas, _ = np.linalg.eig(temp_K)
    S_alpha = np.log2( np.sum( (lambdas.real)**alpha_val ) ) /(1.0 - alpha_val)
    return S_alpha
