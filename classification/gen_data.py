import sys
import os, pickle
from random import shuffle, randint
import gzip, csv
#from types import ListType

from scipy.io import savemat
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import skimage
import matplotlib.image as mpimg


def shuffle_data_x_and_y(data_x, data_y):
    num_x, num_y = data_x.shape[0], data_y.shape[0]
    assert (num_x == num_y), "Given two data have different number of data points"

    indices = list(range(num_x))
    shuffle(indices)
    new_data_x, new_data_y = np.array(data_x[indices]), np.array(data_y[indices])
    return new_data_x, new_data_y

#### function to split data into each categories (gather data of same digit)
def data_class_split(raw_data_x_and_y, num_class):
    raw_x, raw_y = raw_data_x_and_y
    img_bins = [[] for _ in range(num_class)]
    for cnt in range(raw_x.shape[0]):
        img_bins[int(raw_y[cnt])].append(raw_x[cnt])
    return img_bins

#### function to split train data into train and validation set
def data_split_for_validation_data(categorized_train_data, num_train_valid=None, ratio_of_valid_to_train=None):
    num_class = len(categorized_train_data)
    if num_train_valid is not None:
        _num_train_data, _num_valid_data = num_train_valid
        num_train_datas, num_valid_datas = [_num_train_data for _ in range(num_class)], [_num_valid_data for _ in range(num_class)]
    elif ratio_of_valid_to_train is not None:
        num_train_datas, num_valid_datas = [], []
        for images_of_each_class in categorized_train_data:
            num_data = images_of_each_class.shape[0] if type(images_of_each_class)==np.ndarray else len(images_of_each_class)
            num_valid_datas.append(int(num_data*ratio_of_valid_to_train/(1.0+ratio_of_valid_to_train)))
            num_train_datas.append(min(num_data-num_valid_datas[-1], int(num_data/(1.0+ratio_of_valid_to_train))))
    else:
        print("Must provide either number of train/valid data or ratio of valid to train!")
        raise ValueError

    train_img, valid_img = [[] for _ in range(num_class)], [[] for _ in range(num_class)]
    for class_cnt, (images_of_each_class, num_train_data, num_valid_data) in enumerate(zip(categorized_train_data, num_train_datas, num_valid_datas)):
        num_data = images_of_each_class.shape[0] if type(images_of_each_class)==np.ndarray else len(images_of_each_class)
        indices = list(range(num_data))
        shuffle(indices)
        for data_cnt in range(num_data):
            if data_cnt < num_valid_data:
                valid_img[class_cnt].append(images_of_each_class[indices[data_cnt]])
            elif data_cnt < num_valid_data + num_train_data:
                train_img[class_cnt].append(images_of_each_class[indices[data_cnt]])
            else:
                break
    return (train_img, valid_img)


# MNIST data (label : number of train/number of valid/number of test)
# 0 : 5444/479/980, 1 : 6179/563/1135, 2 : 5470/488/1032, 3 : 5638/493/1010
# 4 : 5307/535/982, 5 : 4987/434/892,  6 : 5417/501/958,  7 : 5715/550/1028
# 8 : 5389/462/974, 9 : 5454/495/1009

#### function to split data into each categories (gather data of same digit)
def mnist_data_class_split(mnist_class):
    train_img = data_class_split((mnist_class.train.images, mnist_class.train.labels), 10)
    valid_img = data_class_split((mnist_class.validation.images, mnist_class.validation.labels), 10)
    test_img = data_class_split((mnist_class.test.images, mnist_class.test.labels), 10)
    return (train_img, valid_img, test_img)

#### function to shuffle and randomly select some portion of given data
def shuffle_select_some_data(list_of_data, ratio_to_choose):
    #### assume list_of_data = [[ndarray, ndarray, ...], [ndarray, ndarray, ...], [], ..., []]
    ####        with number of data points for ndarray, and number of classes for list
    if ratio_to_choose > 1.0:
        ratio_to_choose = float(ratio_to_choose)/100.0

    selected_list_of_data = []
    for class_cnt in range(len(list_of_data)):
        num_data_in_this_class = len(list_of_data[class_cnt])
        num_data_to_choose = int(ratio_to_choose * num_data_in_this_class)

        data_copy = list(list_of_data[class_cnt])
        shuffle(data_copy)
        selected_list_of_data.append(list(data_copy[0:num_data_to_choose]))
    return selected_list_of_data

#### function to concatenate data other than that for the specified class
def concat_data_of_classes(given_data, class_not_to_add):
    data_to_return = []
    for cnt in range(len(given_data)):
        if not (cnt == class_not_to_add):
            data_to_return = data_to_return + given_data[cnt]
    shuffle(data_to_return)
    return data_to_return

#### function to make dataset (either train/valid/test) for binary classification
def mnist_data_gen_binary_classification(img_for_true, img_for_false, dataset_size):
    #### dataset has at least 'min_num_from_each' numbers of instances from each class
    #### thus, the number of data for a class is [min_num_from_each, min_num_from_each + num_variable_class]
    half_of_dataset_size = dataset_size // 2
    if len(img_for_true) < half_of_dataset_size and len(img_for_false) < half_of_dataset_size:
        return (None, None)
    elif len(img_for_true) < half_of_dataset_size:
        num_true = len(img_for_true)
    elif len(img_for_false) < half_of_dataset_size:
        num_true = dataset_size - len(img_for_false)
    else:
        num_true = half_of_dataset_size

    indices_for_true, indices_for_false = list(range(len(img_for_true))), list(range(len(img_for_false)))
    shuffle(indices_for_true)
    shuffle(indices_for_false)

    indices_classes = [1 for _ in range(num_true)] + [0 for _ in range(dataset_size-num_true)]
    shuffle(indices_classes)

    data_x, data_y, cnt_true = [], [], 0
    for cnt in range(dataset_size):
        if indices_classes[cnt] == 1:
            data_x.append(img_for_true[indices_for_true[cnt_true]])
            data_y.append(1)
            cnt_true = cnt_true+1
        else:
            data_x.append(img_for_false[indices_for_false[cnt - cnt_true]])
            data_y.append(0)
    return (data_x, data_y)

#### function to print information of data file (number of parameters, dimension, etc.)
def mnist_data_print_info(train_data, valid_data, test_data, no_group=False, print_info=True):
    if no_group:
        num_task = len(train_data)

        num_train, num_valid, num_test = [train_data[x][0].shape[0] for x in range(num_task)], [valid_data[x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0].shape[1], 0
        y_depth = [int(np.amax(x[1])+1) for x in train_data]
        if print_info:
            print("Tasks : %d\nTrain data : %d, Validation data : %d, Test data : %d" %(num_task, num_train, num_valid, num_test))
            print("Input dim : %d, Label dim : %d" %(x_dim, y_dim))
            print("Maximum label : ", y_depth, "\n")
        return (num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth)
    else:
        assert (len(train_data) == len(valid_data)), "Different number of groups in train/validation data"
        num_group = len(train_data)

        bool_num_task = [(len(train_data[0]) == len(train_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in train data"
        bool_num_task = [(len(valid_data[0]) == len(valid_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in validation data"
        assert (len(train_data[0])==len(valid_data[0]) and len(train_data[0])==len(test_data)), "Different number of tasks in train/validation/test data"
        num_task = len(train_data[0])

        num_train, num_valid, num_test = [train_data[0][x][0].shape[0] for x in range(num_task)], [valid_data[0][x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0][0].shape[1], 0
        y_depth = [int(np.amax(x[1])+1) for x in train_data[0]]
        if print_info:
            print("Tasks : %d, Groups of training/valid : %d\n" %(num_task, num_group))
            print("Train data : ", num_train, ", Validation data : ", num_valid, ", Test data : ", num_test)
            print("Input dim : %d, Label dim : %d" %(x_dim, y_dim))
            print("Maximum label : ", y_depth, "\n")
        return (num_task, num_group, num_train, num_valid, num_test, x_dim, y_dim, y_depth)


#### generate/handle data of mnist
#### data format (train_data, validation_data, test_data)
####    - train/validation : [group1(list), group2(list), ... ] with the group of test data format
####    - test : [(task1_x, task1_y), (task2_x, task2_y), ... ]
def mnist_data(data_file_name, num_train_max, num_valid_max, num_test_max, num_train_group, num_tasks=5, data_percent=100):
    curr_path = os.getcwd()
    if not ('Data' in os.listdir(curr_path)):
        os.mkdir('./Data')

    data_path = curr_path + '/Data'
    if data_file_name in os.listdir(data_path):
        with open('./Data/' + data_file_name, 'rb') as fobj:
            if sys.version_info.major < 3:
                train_data, validation_data, test_data = pickle.load(fobj)
            else:
                train_data, validation_data, test_data = pickle.load(fobj, encoding='latin1')
            print('Successfully load data')
    else:
        mnist = input_data.read_data_sets('Data/MNIST_data', one_hot=False)
        #### subclasses : train, validation, test with images/labels subclasses
        categorized_train_x, categorized_valid_x, categorized_test_x = mnist_data_class_split(mnist)

        if num_tasks == 5:
            #### split data into completely different multi-task datasets (5 tasks of 0 vs 1, 2 vs 3, ..., 8 vs 9)
            ## process train data
            train_data = []
            for group_cnt in range(num_train_group):
                train_data_tmp = []
                for task_cnt in range(num_tasks):
                    train_x_tmp, train_y_tmp = mnist_data_gen_binary_classification(categorized_train_x[2*task_cnt], categorized_train_x[2*task_cnt+1], num_train_max)
                    train_data_tmp.append( ( np.array(train_x_tmp), np.array(train_y_tmp) ) )
                train_data.append(train_data_tmp)

            ## process validation data
            validation_data = []
            for group_cnt in range(num_train_group):
                validation_data_tmp = []
                for task_cnt in range(num_tasks):
                    valid_x_tmp, valid_y_tmp = mnist_data_gen_binary_classification(categorized_valid_x[2*task_cnt], categorized_valid_x[2*task_cnt+1], num_valid_max)
                    validation_data_tmp.append( ( np.array(valid_x_tmp), np.array(valid_y_tmp) ) )
                validation_data.append(validation_data_tmp)

            ## process test data
            test_data = []
            for task_cnt in range(num_tasks):
                test_x_tmp, test_y_tmp = mnist_data_gen_binary_classification(categorized_test_x[2*task_cnt], categorized_test_x[2*task_cnt+1], num_test_max)
                test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )
        elif num_tasks == 10:
            #### split data into 10 tasks of 1-vs-all
            ## process train data
            train_data = []
            for group_cnt in range(num_train_group):
                train_data_tmp = []
                selected_categorized_train_x = shuffle_select_some_data(categorized_train_x, data_percent)

                for task_cnt in range(num_tasks):
                    train_x_tmp, train_y_tmp = mnist_data_gen_binary_classification(selected_categorized_train_x[task_cnt], concat_data_of_classes(selected_categorized_train_x, task_cnt), num_train_max)
                    train_data_tmp.append( ( np.array(train_x_tmp), np.array(train_y_tmp) ) )
                train_data.append(train_data_tmp)

            ## process validation data
            validation_data = []
            for group_cnt in range(num_train_group):
                validation_data_tmp = []
                selected_categorized_valid_x = shuffle_select_some_data(categorized_valid_x, data_percent)
                for task_cnt in range(num_tasks):
                    valid_x_tmp, valid_y_tmp = mnist_data_gen_binary_classification(selected_categorized_valid_x[task_cnt], concat_data_of_classes(selected_categorized_valid_x, task_cnt), num_valid_max)
                    validation_data_tmp.append( ( np.array(valid_x_tmp), np.array(valid_y_tmp) ) )
                validation_data.append(validation_data_tmp)

            ## process test data
            test_data = []
            for task_cnt in range(num_tasks):
                test_x_tmp, test_y_tmp = mnist_data_gen_binary_classification(categorized_test_x[task_cnt], concat_data_of_classes(categorized_test_x, task_cnt), num_test_max)
                test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )
        else:
            print('Check number of tasks - MNIST Data gen.')
            return None

        #### save data
        with open('./Data/' + data_file_name, 'wb') as fobj:
            pickle.dump([train_data, validation_data, test_data], fobj)
            print('Successfully generate/save data')
    return (train_data, validation_data, test_data)



############################################################
#### CIFAR dataset
############################################################
#### read raw data from cifar10 files
def read_cifar10_data(data_path):
    train_x_list, train_y_list = [], []
    for cnt in range(1, 6):
        file_name = 'data_batch_'+str(cnt)
        with open(data_path+'/'+file_name, 'rb') as fobj:
            if sys.version_info.major < 3:
                dict_tmp = pickle.load(fobj)
            else:
                dict_tmp = pickle.load(fobj, encoding='latin1')
            train_x_tmp, train_y_tmp = dict_tmp['data'], dict_tmp['labels']
            num_data_in_file = train_x_tmp.shape[0]
            train_x = train_x_tmp.reshape(num_data_in_file, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
            train_x = train_x.reshape(num_data_in_file, 3*32*32)
            train_y = np.array(train_y_tmp)

            train_x_list.append(train_x)
            train_y_list.append(train_y)
            print("\tRead %s file" %(file_name))
    proc_train_x, proc_train_y = np.concatenate(train_x_list), np.concatenate(train_y_list)

    with open(data_path+'/test_batch', 'rb') as fobj:
        if sys.version_info.major < 3:
            dict_tmp = pickle.load(fobj)
        else:
            dict_tmp = pickle.load(fobj, encoding='latin1')
        test_x_tmp, test_y_tmp = dict_tmp['data'], dict_tmp['labels']
        num_data_in_file = test_x_tmp.shape[0]
        test_x = test_x_tmp.reshape(num_data_in_file, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
        proc_test_x = test_x.reshape(num_data_in_file, 3*32*32).astype(np.float32)
        proc_test_y = np.array(test_y_tmp)
        print("\tRead test_batch file")
    return (proc_train_x, proc_train_y, proc_test_x, proc_test_y)


#### function to normalize cifar data
def cifar_data_standardization(raw_data):
    if len(raw_data.shape)<2:
        ## single numpy array
        num_elem = raw_data.shape[0]
        raw_data_reshaped = raw_data.reshape(num_elem//3, 3)
        mean, std = np.mean(raw_data_reshaped, dtype=np.float32, axis=0), np.std(raw_data_reshaped, dtype=np.float32, axis=0)
        adjusted_std = np.maximum(std, [1.0/np.sqrt(raw_data_reshaped.shape[0]) for _ in range(3)])
        new_data = ((raw_data_reshaped - mean)/adjusted_std).reshape(num_elem)
    else:
        ## 2D numpy array
        num_data, num_feature = raw_data.shape
        raw_data_reshaped = raw_data.reshape(num_data*num_feature//3, 3)
        mean, std = np.mean(raw_data_reshaped, dtype=np.float32, axis=0), np.std(raw_data_reshaped, dtype=np.float32, axis=0)
        adjusted_std = np.maximum(std, [1.0/np.sqrt(raw_data_reshaped.shape[0]) for _ in range(3)])
        new_data = ((raw_data_reshaped - mean)/adjusted_std).reshape(num_data, num_feature)
    return new_data


#### function to make dataset (either train/valid/test) for binary classification
def cifar_data_gen_binary_classification(img_for_true, img_for_false, dataset_size):
    #### dataset has at least 'min_num_from_each' numbers of instances from each class
    #### thus, the number of data for a class is [min_num_from_each, min_num_from_each + num_variable_class]
    if dataset_size < 1:
        dataset_size = len(img_for_true) + len(img_for_false)

    num_true = min(dataset_size//2, len(img_for_true))

    indices_for_true, indices_for_false = list(range(len(img_for_true))), list(range(len(img_for_false)))
    shuffle(indices_for_true)
    shuffle(indices_for_false)

    indices_classes = [1 for _ in range(num_true)] + [0 for _ in range(dataset_size-num_true)]
    shuffle(indices_classes)

    data_x, data_y, cnt_false = [], [], 0
    for cnt in range(dataset_size):
        if indices_classes[cnt] == 0:
            img_tmp = img_for_false[indices_for_false[cnt_false]]
            data_x.append(img_tmp)
            data_y.append(0)
            cnt_false = cnt_false+1
        else:
            img_tmp = img_for_true[indices_for_true[cnt-cnt_false]]
            data_x.append(img_tmp)
            data_y.append(1)
    data_x = cifar_data_standardization(np.array(data_x))
    return (data_x, data_y)


#### function to make dataset (either train/valid/test) for binary classification
def cifar_data_gen_multiclass_classification(imgs, num_class, dataset_size):
    #### dataset has at least 'min_num_from_each' numbers of instances from each class
    #### thus, the number of data for a class is [min_num_from_each, min_num_from_each + num_variable_class]
    if dataset_size < 1:
        dataset_size = sum([len(x) for x in imgs])

    num_at_each_class = [int(dataset_size/num_class) for _ in range(num_class)]

    indices_classes, indices_imgs = [], [list(range(len(x))) for x in imgs]
    for cnt in range(num_class):
        indices_classes = indices_classes + [cnt for _ in range(num_at_each_class[cnt])]
        shuffle(indices_imgs[cnt])
    shuffle(indices_classes)

    data_x, data_y, cnt_imgs = [], [], [0 for _ in range(num_class-1)]
    for cnt in range(dataset_size):
        img_label = indices_classes[cnt]
        if img_label > num_class-2:
            img_tmp = imgs[img_label][cnt-sum(cnt_imgs)]
        else:        
            img_tmp = imgs[img_label][cnt_imgs[img_label]]
            cnt_imgs[img_label] = cnt_imgs[img_label]+1
        data_x.append(img_tmp)
        data_y.append(img_label)
    data_x = cifar_data_standardization(np.array(data_x))
    return (data_x, data_y)


#### function to print information of data file (number of parameters, dimension, etc.)
def cifar_data_print_info(train_data, valid_data, test_data, no_group=False, print_info=True, grouped_test=False):
    if no_group:
        num_task = len(train_data)

        num_train, num_valid, num_test = [train_data[x][0].shape[0] for x in range(num_task)], [valid_data[x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0].shape[1], 0
        y_depth = [int(np.amax(x[1])+1) for x in train_data]
        if print_info:
            print("Tasks : %d\nTrain data : %d, Validation data : %d, Test data : %d" %(num_task, num_train, num_valid, num_test))
            print("Input dim : %d, Label dim : %d" %(x_dim, y_dim))
            print("Maximum label : ", y_depth, "\n")
        return (num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth)
    else:
        if grouped_test:
            assert (len(train_data) == len(valid_data) and len(valid_data) == len(test_data)), "Different number of groups in train/validation data"
            assert (len(train_data[0])==len(valid_data[0]) and len(train_data[0])==len(test_data[0])), "Different number of tasks in train/validation/test data"
        else:
            assert (len(train_data) == len(valid_data)), "Different number of groups in train/validation data"
            assert (len(train_data[0])==len(valid_data[0]) and len(train_data[0])==len(test_data)), "Different number of tasks in train/validation/test data"
        num_group = len(train_data)

        bool_num_task = [(len(train_data[0]) == len(train_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in train data"
        bool_num_task = [(len(valid_data[0]) == len(valid_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in validation data"
        if grouped_test:
            bool_num_task = [(len(test_data[0]) == len(test_data[x])) for x in range(1, num_group)]
            assert all(bool_num_task), "Different number of tasks in some of groups in test data"
        num_task = len(train_data[0])

        num_train, num_valid = [train_data[0][x][0].shape[0] for x in range(num_task)], [valid_data[0][x][0].shape[0] for x in range(num_task)]
        if grouped_test:
            num_test = [test_data[0][x][0].shape[0] for x in range(num_task)]
        else:
            num_test = [test_data[x][0].shape[0] for x in range(num_task)]

        x_dim, y_dim = train_data[0][0][0].shape[1], 0
        y_depth = [int(np.amax(x[1])+1) for x in train_data[0]]
        if print_info:
            print("Tasks : %d, Groups of training/valid : %d\n" %(num_task, num_group))
            print("Train data : ", num_train, ", Validation data : ", num_valid, ", Test data : ", num_test)
            print("Input dim : %d, Label dim : %d" %(x_dim, y_dim))
            print("Maximum label : ", y_depth, "\n")
        return (num_task, num_group, num_train, num_valid, num_test, x_dim, y_dim, y_depth)


#### process cifar10 data to generate pickle file with data in right format for DNN models
#### train/test data : 5000/1000 per class
def cifar10_data(data_file_name, num_train_max, num_valid_max, num_test_max, num_train_group, num_tasks, train_valid_ratio=0.2, multiclass=False, save_as_mat=False, data_percent=100):
    curr_path = os.getcwd()
    if not ('Data' in os.listdir(curr_path)):
        os.mkdir('./Data')

    data_path = curr_path + '/Data'
    if data_file_name in os.listdir(data_path):
        with open('./Data/' + data_file_name, 'rb') as fobj:
            if sys.version_info.major < 3:
                train_data, validation_data, test_data = pickle.load(fobj)
            else:
                train_data, validation_data, test_data = pickle.load(fobj, encoding='latin1')
            print('Successfully load data')
    else:
        raw_data_path = curr_path + '/Data/cifar-10-batches-py'
        cifar10_trainx, cifar10_trainy, cifar10_testx, cifar10_testy = read_cifar10_data(raw_data_path)

        #### split data into sets for each label
        categorized_train_x_tmp = data_class_split([cifar10_trainx, cifar10_trainy], 10)
        categorized_test_x = data_class_split([cifar10_testx, cifar10_testy], 10)
        categorized_train_x, categorized_valid_x = data_split_for_validation_data(categorized_train_x_tmp, ratio_of_valid_to_train=train_valid_ratio)

        if multiclass:
            #### make data into multi-class
            ## process train data
            train_data = []
            for group_cnt in range(num_train_group):
                train_x_tmp, train_y_tmp = cifar_data_gen_multiclass_classification(categorized_train_x, 10, num_train_max)
                train_data.append( [( np.array(train_x_tmp), np.array(train_y_tmp) )] )

            ## process validation data
            validation_data = []
            for group_cnt in range(num_train_group):
                valid_x_tmp, valid_y_tmp = cifar_data_gen_multiclass_classification(categorized_valid_x, 10, num_valid_max)
                validation_data.append( [( np.array(valid_x_tmp), np.array(valid_y_tmp) )] )

            ## process test data
            test_data = []
            for group_cnt in range(num_train_group):
                test_x_tmp, test_y_tmp = cifar_data_gen_multiclass_classification(categorized_test_x, 10, num_test_max)
                test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )
        else:
            if num_tasks == 5:
                #### split data into completely different multi-task datasets (5 tasks of 0 vs 1, 2 vs 3, ..., 8 vs 9)
                ## process train data
                train_data = []
                for group_cnt in range(num_train_group):
                    train_data_tmp = []
                    for task_cnt in range(num_tasks):
                        train_x_tmp, train_y_tmp = cifar_data_gen_binary_classification(categorized_train_x[2*task_cnt], categorized_train_x[2*task_cnt+1], num_train_max)
                        train_data_tmp.append( ( np.array(train_x_tmp), np.array(train_y_tmp) ) )
                    train_data.append(train_data_tmp)

                ## process validation data
                validation_data = []
                for group_cnt in range(num_train_group):
                    validation_data_tmp = []
                    for task_cnt in range(num_tasks):
                        valid_x_tmp, valid_y_tmp = cifar_data_gen_binary_classification(categorized_valid_x[2*task_cnt], categorized_valid_x[2*task_cnt+1], num_valid_max)
                        validation_data_tmp.append( ( np.array(valid_x_tmp), np.array(valid_y_tmp) ) )
                    validation_data.append(validation_data_tmp)

                ## process test data
                test_data = []
                for task_cnt in range(num_tasks):
                    test_x_tmp, test_y_tmp = cifar_data_gen_binary_classification(categorized_test_x[2*task_cnt], categorized_test_x[2*task_cnt+1], num_test_max)
                    test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )

            elif num_tasks == 10:
                #### split data into 10 tasks of one-vs-all multi-task datasets
                ## process train data
                train_data = []
                for group_cnt in range(num_train_group):
                    train_data_tmp = []
                    selected_categorized_train_x = shuffle_select_some_data(categorized_train_x, data_percent)

                    for task_cnt in range(num_tasks):
                        train_x_tmp, train_y_tmp = cifar_data_gen_binary_classification(selected_categorized_train_x[task_cnt], concat_data_of_classes(selected_categorized_train_x, task_cnt), num_train_max)
                        train_data_tmp.append( ( np.array(train_x_tmp), np.array(train_y_tmp) ) )
                    train_data.append(train_data_tmp)

                ## process validation data
                validation_data = []
                for group_cnt in range(num_train_group):
                    validation_data_tmp = []
                    selected_categorized_valid_x = shuffle_select_some_data(categorized_valid_x, data_percent)

                    for task_cnt in range(num_tasks):
                        valid_x_tmp, valid_y_tmp = cifar_data_gen_binary_classification(selected_categorized_valid_x[task_cnt], concat_data_of_classes(selected_categorized_valid_x, task_cnt), num_valid_max)
                        validation_data_tmp.append( ( np.array(valid_x_tmp), np.array(valid_y_tmp) ) )
                    validation_data.append(validation_data_tmp)

                ## process test data
                test_data = []
                for task_cnt in range(num_tasks):
                    test_x_tmp, test_y_tmp = cifar_data_gen_binary_classification(categorized_test_x[task_cnt], concat_data_of_classes(categorized_test_x, task_cnt), num_test_max)
                    test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )

        #### save data
        if save_as_mat:
            data_to_save_in_mat = _data_save_to_mat(train_data, validation_data, test_data, num_train_group, num_tasks)
            savemat('./Data/'+data_file_name[0:-4]+'.mat', data_to_save_in_mat)

        with open('./Data/' + data_file_name, 'wb') as fobj:
            pickle.dump([train_data, validation_data, test_data], fobj)
            print('Successfully generate/save data')
    return (train_data, validation_data, test_data)



################################# CIFAR-100 Data
# [first five tasks]
# 'aquatic_mammals'			4, 30, 55, 72, 95
# 'fish'					1, 32, 67, 73, 91
# 'flowers'				54, 62, 70, 82, 92
# 'food_containers'			9, 10, 16, 28, 61
# 'fruit_and_vegetables'			0, 51, 53, 57, 83
# 'household_furniture'			5, 20, 25, 84, 94
# 'insects'				6, 7, 14, 18, 24
# 'large_carnivores'			3, 42, 43, 88, 97
# 'large_man-made_outdoor_things'		12, 17, 37, 68, 76
# 'vehicles_1'				8, 13, 48, 58, 90
# 
# [second five tasks]
# 'household_electrical_devices'		22, 39, 40, 86, 87
# 'large_natural_outdoor_scenes'		23, 33, 49, 60, 71
# 'large_omnivores_and_herbivores'	15, 19, 21, 31, 38
# 'medium_mammals'			34, 63, 64, 66, 75
# 'non-insect_invertebrates'		26, 45, 77, 79, 99
# 'people'				2, 11, 35, 46, 98
# 'reptiles'				27, 29, 44, 78, 93
# 'small_mammals'				36, 50, 65, 74, 80
# 'trees'					47, 52, 56, 59, 96
# 'vehicles_2'				41, 69, 81, 85, 89

_cifar100_task_labels_10 = [[4, 1, 54, 9, 0, 5, 6, 3, 12, 8],
                            [30, 32, 62, 10, 51, 20, 7, 42, 17, 13],
                            [55, 67, 70, 16, 53, 25, 14, 43, 37, 48],
                            [72, 73, 82, 28, 57, 84, 18, 88, 68, 58],
                            [95, 91, 92, 61, 83, 94, 24, 97, 76, 90],
                            [22, 23, 15, 34, 26, 2, 27, 36, 47, 41],
                            [39, 33, 19, 63, 45, 11, 29, 50, 52, 69],
                            [40, 49, 21, 64, 77, 35, 44, 65, 56, 81],
                            [86, 60, 31, 66, 79, 46, 78, 74, 59, 85],
                            [87, 71, 38, 75, 99, 98, 93, 80, 96, 89]]

_cifar100_task_labels_20 = [[4, 1, 54, 9, 0, 5, 6, 3, 12, 8],
                            [4, 23, 54, 34, 0, 2, 6, 36, 12, 41],
                            [22, 23, 15, 34, 26, 2, 27, 36, 47, 41],
                            [22, 32, 15, 10, 26, 20, 27, 42, 47, 13],
                            [30, 32, 62, 10, 51, 20, 7, 42, 17, 13],
                            [30, 33, 62, 63, 51, 11, 7, 50, 17, 69],
                            [39, 33, 19, 63, 45, 11, 29, 50, 52, 69],
                            [39, 67, 19, 16, 45, 25, 29, 43, 52, 48],
                            [55, 67, 70, 16, 53, 25, 14, 43, 37, 48],
                            [55, 49, 70, 64, 53, 35, 14, 65, 37, 81],
                            [40, 49, 21, 64, 77, 35, 44, 65, 56, 81],
                            [40, 73, 21, 28, 77, 84, 44, 88, 56, 58],
                            [72, 73, 82, 28, 57, 84, 18, 88, 68, 58],
                            [72, 60, 82, 66, 57, 46, 18, 74, 68, 85],
                            [86, 60, 31, 66, 79, 46, 78, 74, 59, 85],
                            [86, 71, 31, 75, 79, 98, 78, 80, 59, 89],
                            [87, 71, 38, 75, 99, 98, 93, 80, 96, 89],
                            [87, 91, 38, 61, 99, 94, 93, 97, 96, 90],
                            [95, 91, 92, 61, 83, 94, 24, 97, 76, 90],
                            [95, 1, 92, 9, 83, 5, 24, 3, 76, 8]]

_cifar100_task_labels_40 = [[4, 1, 54, 9, 0, 5, 6, 3, 12, 8],
                            [4, 23, 54, 34, 0, 2, 6, 36, 12, 41],
                            [22, 23, 15, 34, 26, 2, 27, 36, 47, 41],
                            [22, 32, 15, 10, 26, 20, 27, 42, 47, 13],
                            [30, 32, 62, 10, 51, 20, 7, 42, 17, 13],
                            [30, 33, 62, 63, 51, 11, 7, 50, 17, 69],
                            [39, 33, 19, 63, 45, 11, 29, 50, 52, 69],
                            [39, 67, 19, 16, 45, 25, 29, 43, 52, 48],
                            [55, 67, 70, 16, 53, 25, 14, 43, 37, 48],
                            [55, 49, 70, 64, 53, 35, 14, 65, 37, 81],
                            [40, 49, 21, 64, 77, 35, 44, 65, 56, 81],
                            [40, 73, 21, 28, 77, 84, 44, 88, 56, 58],
                            [72, 73, 82, 28, 57, 84, 18, 88, 68, 58],
                            [72, 60, 82, 66, 57, 46, 18, 74, 68, 85],
                            [86, 60, 31, 66, 79, 46, 78, 74, 59, 85],
                            [86, 71, 31, 75, 79, 98, 78, 80, 59, 89],
                            [87, 71, 38, 75, 99, 98, 93, 80, 96, 89],
                            [87, 91, 38, 61, 99, 94, 93, 97, 96, 90],
                            [95, 91, 92, 61, 83, 94, 24, 97, 76, 90],
                            [95, 1, 92, 9, 83, 5, 24, 3, 76, 8],
                            [4, 22, 54, 15, 0, 26, 6, 27, 12, 47],
                            [1, 23, 9, 34, 5, 26, 6, 27, 12, 47],
                            [1, 23, 9, 34, 5, 2, 3, 36, 8, 41],
                            [30, 39, 62, 19, 51, 2, 3, 36, 8, 41],
                            [30, 39, 62, 19, 51, 45, 7, 29, 17, 52],
                            [32, 33, 10, 63, 20, 45, 7, 29, 17, 52],
                            [32, 33, 10, 63, 20, 11, 42, 50, 13, 69],
                            [55, 40, 70, 21, 53, 11, 42, 50, 13, 69],
                            [55, 40, 70, 21, 53, 77, 14, 44, 37, 56],
                            [67, 49, 16, 64, 25, 77, 14, 44, 37, 56],
                            [67, 49, 16, 64, 25, 35, 43, 65, 48, 81],
                            [72, 86, 82, 31, 57, 35, 43, 65, 48, 81],
                            [72, 86, 82, 31, 57, 79, 18, 78, 68, 59],
                            [73, 60, 28, 66, 84, 79, 18, 78, 68, 59],
                            [73, 60, 28, 66, 84, 46, 88, 74, 58, 85],
                            [95, 87, 92, 38, 83, 46, 88, 74, 58, 85],
                            [95, 87, 92, 38, 83, 99, 24, 93, 76, 96],
                            [91, 71, 61, 75, 94, 99, 24, 93, 76, 96],
                            [91, 71, 61, 75, 94, 98, 97, 80, 90, 89],
                            [4, 22, 54, 15, 0, 98, 97, 80, 90, 89]]

#### read raw data from cifar100 files
def read_cifar100_data(data_path):
    with open(data_path+'/train', 'rb') as fobj:
        if sys.version_info.major < 3:
            dict_tmp = pickle.load(fobj)
        else:
            dict_tmp = pickle.load(fobj, encoding='latin1')
        train_x_tmp, train_y_tmp = dict_tmp['data'], dict_tmp['fine_labels']
        num_data_in_file = train_x_tmp.shape[0]
        train_x = train_x_tmp.reshape(num_data_in_file, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
        proc_train_x = train_x.reshape(num_data_in_file, 3*32*32).astype(np.float32)
        proc_train_y = np.array(train_y_tmp)
        print("\tRead train_batch file")

    with open(data_path+'/test', 'rb') as fobj:
        if sys.version_info.major < 3:
            dict_tmp = pickle.load(fobj)
        else:
            dict_tmp = pickle.load(fobj, encoding='latin1')
        test_x_tmp, test_y_tmp = dict_tmp['data'], dict_tmp['fine_labels']
        num_data_in_file = test_x_tmp.shape[0]
        test_x = test_x_tmp.reshape(num_data_in_file, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32) / 255.0
        proc_test_x = test_x.reshape(num_data_in_file, 3*32*32).astype(np.float32)
        proc_test_y = np.array(test_y_tmp)
        print("\tRead test_batch file")
    return (proc_train_x, proc_train_y, proc_test_x, proc_test_y)


#### process cifar100 data to generate pickle file with data in right format for DNN models
#### train/test data : 500/100 per class
def cifar100_data(data_file_name, num_train_max, num_valid_max, num_test_max, num_train_group, num_tasks, train_valid_ratio=0.2, multiclass=False, save_as_mat=False):
    curr_path = os.getcwd()
    if not ('Data' in os.listdir(curr_path)):
        os.mkdir('./Data')

    data_path = curr_path + '/Data'
    if data_file_name in os.listdir(data_path):
        with open('./Data/' + data_file_name, 'rb') as fobj:
            if sys.version_info.major < 3:
                train_data, validation_data, test_data = pickle.load(fobj)
            else:
                train_data, validation_data, test_data = pickle.load(fobj, encoding='latin1')
            print('Successfully load data')
    else:
        raw_data_path = curr_path + '/Data/cifar-100-python'
        cifar100_trainx, cifar100_trainy, cifar100_testx, cifar100_testy = read_cifar100_data(raw_data_path)

        #### split data into sets for each label
        categorized_train_x_tmp = data_class_split([cifar100_trainx, cifar100_trainy], 100)
        categorized_test_x = data_class_split([cifar100_testx, cifar100_testy], 100)
        categorized_train_x, categorized_valid_x = data_split_for_validation_data(categorized_train_x_tmp, ratio_of_valid_to_train=train_valid_ratio)

        if multiclass:
            #### make data into multi-class
            ## process train data
            assert (num_tasks == len(_cifar100_task_labels_10) or num_tasks == len(_cifar100_task_labels_20) or num_tasks == len(_cifar100_task_labels_40)), "Number of tasks doesn't match the group of class labels"

            if num_tasks == len(_cifar100_task_labels_10):
                _cifar100_task_labels = _cifar100_task_labels_10
            elif num_tasks == len(_cifar100_task_labels_20):
                _cifar100_task_labels = _cifar100_task_labels_20
            elif num_tasks == len(_cifar100_task_labels_40):
                _cifar100_task_labels = _cifar100_task_labels_40

            train_data = []
            for group_cnt in range(num_train_group):
                train_data_tmp = []
                for task_cnt in range(num_tasks):
                    train_x_tmp, train_y_tmp = cifar_data_gen_multiclass_classification([categorized_train_x[x] for x in _cifar100_task_labels[task_cnt]], len(_cifar100_task_labels[task_cnt]), num_train_max)
                    train_data_tmp.append( ( np.array(train_x_tmp), np.array(train_y_tmp) ) )
                train_data.append(train_data_tmp)

            ## process validation data
            validation_data = []
            for group_cnt in range(num_train_group):
                validation_data_tmp = []
                for task_cnt in range(num_tasks):
                    valid_x_tmp, valid_y_tmp = cifar_data_gen_multiclass_classification([categorized_valid_x[x] for x in _cifar100_task_labels[task_cnt]], len(_cifar100_task_labels[task_cnt]), num_valid_max)
                    validation_data_tmp.append( ( np.array(valid_x_tmp), np.array(valid_y_tmp) ) )
                validation_data.append(validation_data_tmp)

            ## process test data
            test_data = []
            for task_cnt in range(num_tasks):
                test_x_tmp, test_y_tmp = cifar_data_gen_multiclass_classification([categorized_test_x[x] for x in _cifar100_task_labels[task_cnt]], len(_cifar100_task_labels[task_cnt]), num_test_max)
                test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )
        else:
            #### split data into completely different multi-task datasets (5 tasks of 0 vs 1, 2 vs 3, ..., 8 vs 9)
            ## process train data
            train_data = []
            for group_cnt in range(num_train_group):
                train_data_tmp = []
                for task_cnt in range(num_tasks):
                    train_x_tmp, train_y_tmp = cifar_data_gen_binary_classification(categorized_train_x[2*task_cnt], categorized_train_x[2*task_cnt+1], num_train_max)
                    train_data_tmp.append( ( np.array(train_x_tmp), np.array(train_y_tmp) ) )
                train_data.append(train_data_tmp)

            ## process validation data
            validation_data = []
            for group_cnt in range(num_train_group):
                validation_data_tmp = []
                for task_cnt in range(num_tasks):
                    valid_x_tmp, valid_y_tmp = cifar_data_gen_binary_classification(categorized_valid_x[2*task_cnt], categorized_valid_x[2*task_cnt+1], num_valid_max)
                    validation_data_tmp.append( ( np.array(valid_x_tmp), np.array(valid_y_tmp) ) )
                validation_data.append(validation_data_tmp)

            ## process test data
            test_data = []
            for task_cnt in range(num_tasks):
                test_x_tmp, test_y_tmp = cifar_data_gen_binary_classification(categorized_test_x[2*task_cnt], categorized_test_x[2*task_cnt+1], num_test_max)
                test_data.append( ( np.array(test_x_tmp), np.array(test_y_tmp) ) )

        #### save data
        if save_as_mat:
            data_to_save_in_mat = _data_save_to_mat(train_data, validation_data, test_data, num_train_group, num_tasks)
            savemat('./Data/'+data_file_name[0:-4]+'.mat', data_to_save_in_mat)

        with open('./Data/' + data_file_name, 'wb') as fobj:
            pickle.dump([train_data, validation_data, test_data], fobj)
            print('Successfully generate/save data')
    return (train_data, validation_data, test_data)


################################### Office-Home Data
# 65 classes
_officehome_class_labels = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar',
                            'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser',
                            'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork', 'Glasses', 'Hammer',
                            'Helmet', 'Kettle', 'Keyboard', 'Knives', 'Lamp_Shade', 'Laptop', 'Marker', 'Monitor', 'Mop',
                            'Mouse', 'Mug', 'Notebook', 'Oven', 'Pan', 'Paper_Clip', 'Pen', 'Pencil', 'Postit_Notes', 'Printer',
                            'Push_Pin', 'Radio', 'Refrigerator', 'Ruler', 'Scissors', 'Screwdriver', 'Shelf', 'Sink', 'Sneakers',
                            'Soda', 'Speaker', 'Spoon', 'Table', 'Telephone', 'ToothBrush', 'Toys', 'Trash_Can', 'TV', 'Webcam']

# randomly split subtasks
_officehome_task_labels  = [ [63, 54, 55, 59, 47, 8, 5, 56, 18, 50, 3, 57, 13],
                             [62, 19, 21, 35, 64, 46, 27, 10, 39, 17, 22, 34, 30],
                             [48, 15, 51, 6, 40, 31, 4, 36, 45, 41, 16, 14, 11],
                             [9, 29, 1, 44, 33, 26, 25, 32, 53, 12, 43, 38, 60],
                             [28, 2, 49, 37, 20, 23, 58, 24, 52, 61, 7, 0, 42] ]


def read_officehome_images(image_type, img_size):
    assert ( (image_type == 'Art') or (image_type == 'Clipart') or (image_type == 'Product') or (image_type == 'RealWorld') ), "Given image type of Office-Home dataset is wrong!"
    raw_images = []
    for class_label in _officehome_class_labels:
        img_path = os.getcwd()+'/Data/OfficeHomeDataset/'+image_type+'/'+class_label+'/'
        num_imgs = len(os.listdir(img_path))
        raw_class_images = np.zeros([num_imgs, img_size[0]*img_size[1]*img_size[2]], dtype=np.float32)
        for img_cnt in range(num_imgs):
            img_name = img_path + format(int(img_cnt+1), '05d') + '.jpg'
            img_tmp = mpimg.imread(img_name).astype(np.float32)/255.0
            img = skimage.transform.resize(img_tmp, img_size, anti_aliasing=True)
            raw_class_images[img_cnt] = np.array(img.reshape(-1))
        raw_images.append(np.array(raw_class_images))
        print("\t\tImage class - %s" %(class_label))
    return raw_images

def read_officehome_all_images(super_class_list, img_size):
    raw_images = {}
    if 'Art' in super_class_list:
        raw_images['Art'] = read_officehome_images('Art', img_size)
        print("\tComplete reading Office-Home/ Art images")
        print([c.shape[0] for c in raw_images['Art']])
        print("\n")

    if 'Clipart' in super_class_list:
        raw_images['Clipart'] = read_officehome_images('Clipart', img_size)
        print("\tComplete reading Office-Home/ Clipart images")
        print([c.shape[0] for c in raw_images['Clipart']])
        print("\n")

    if 'Product' in super_class_list:
        raw_images['Product'] = read_officehome_images('Product', img_size)
        print("\tComplete reading Office-Home/ Product images")
        print([c.shape[0] for c in raw_images['Product']])
        print("\n")

    if 'RealWorld' in super_class_list:
        raw_images['RealWorld'] = read_officehome_images('RealWorld', img_size)
        print("\tComplete reading Office-Home/ RealWorld images")
        print([c.shape[0] for c in raw_images['RealWorld']])
        print("\n")
    return raw_images


def officehome_data(data_file_name, num_train_ratio, num_valid_ratio, num_test_ratio, num_train_group, img_size, save_as_mat=False):
    assert (num_train_ratio + num_valid_ratio + num_test_ratio <= 1.0), "Sum of the given ratio of data should be less than or equal to 1"

    curr_path = os.getcwd()
    if not ('Data' in os.listdir(curr_path)):
        os.mkdir('./Data')

    data_path = curr_path + '/Data'
    if data_file_name in os.listdir(data_path):
        with open('./Data/' + data_file_name, 'rb') as fobj:
            if sys.version_info.major < 3:
                train_data, validation_data, test_data = pickle.load(fobj)
            else:
                train_data, validation_data, test_data = pickle.load(fobj, encoding='latin1')
            print('Successfully load data')
    else:
        data_types = ['Product', 'RealWorld']
        raw_images = read_officehome_all_images(data_types, img_size)

        train_data, validation_data, test_data = [], [], []
        for group_cnt in range(num_train_group):
            train_data_tmp, validation_data_tmp, test_data_tmp = [], [], []

            for data_type in data_types:
                for task_cnt, (task_list) in enumerate(_officehome_task_labels):
                    for class_cnt, (class_label) in enumerate(task_list):
                        imgs_in_class_subtask = raw_images[data_type][class_label]
                        num_imgs = imgs_in_class_subtask.shape[0]
                        num_train, num_valid, num_test = int(num_imgs*num_train_ratio), int(num_imgs*num_valid_ratio), int(num_imgs*num_test_ratio)
                        img_indices = list(range(num_imgs))
                        shuffle(img_indices)

                        train_x_tmp, train_y_tmp = imgs_in_class_subtask[img_indices[0:num_train],:], np.ones(num_train, dtype=np.int32)*class_cnt
                        valid_x_tmp, valid_y_tmp = imgs_in_class_subtask[img_indices[num_train:num_train+num_valid],:], np.ones(num_valid, dtype=np.int32)*class_cnt
                        test_x_tmp, test_y_tmp = imgs_in_class_subtask[img_indices[num_train+num_valid:num_train+num_valid+num_test],:], np.ones(num_test, dtype=np.int32)*class_cnt

                        if class_cnt < 1:
                            train_x, train_y = train_x_tmp, train_y_tmp
                            valid_x, valid_y = valid_x_tmp, valid_y_tmp
                            test_x, test_y = test_x_tmp, test_y_tmp
                        else:
                            train_x, train_y = np.concatenate((train_x, train_x_tmp), axis=0), np.concatenate((train_y, train_y_tmp), axis=0)
                            valid_x, valid_y = np.concatenate((valid_x, valid_x_tmp), axis=0), np.concatenate((valid_y, valid_y_tmp), axis=0)
                            test_x, test_y = np.concatenate((test_x, test_x_tmp), axis=0), np.concatenate((test_y, test_y_tmp), axis=0)

                    train_x, train_y = shuffle_data_x_and_y(train_x, train_y)
                    valid_x, valid_y = shuffle_data_x_and_y(valid_x, valid_y)
                    test_x, test_y = shuffle_data_x_and_y(test_x, test_y)

                    train_data_tmp.append( ( np.array(train_x), np.array(train_y) ) )
                    validation_data_tmp.append( ( np.array(valid_x), np.array(valid_y) ) )
                    test_data_tmp.append( ( np.array(test_x), np.array(test_y) ) )

            train_data.append(train_data_tmp)
            validation_data.append(validation_data_tmp)
            test_data.append(test_data_tmp)
            num_tasks = len(train_data_tmp)

        #### save data
        if save_as_mat:
            data_to_save_in_mat = _data_save_to_mat(train_data, validation_data, test_data, num_train_group, num_tasks)
            savemat('./Data/'+data_file_name[0:-4]+'.mat', data_to_save_in_mat)

        with open('./Data/' + data_file_name, 'wb') as fobj:
            pickle.dump([train_data, validation_data, test_data], fobj)
            print('Successfully generate/save data')
    return (train_data, validation_data, test_data)

def officehome_data_print_info(train_data, valid_data, test_data, no_group=False, print_info=True):
    if no_group:
        num_task = len(train_data)

        num_train, num_valid, num_test = [train_data[x][0].shape[0] for x in range(num_task)], [valid_data[x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0].shape[1], 0
        y_depth = [int(np.amax(x[1])+1) for x in train_data]
        if print_info:
            print("Tasks : %d\nTrain data : %d, Validation data : %d, Test data : %d" %(num_task, num_train, num_valid, num_test))
            print("Input dim : %d, Label dim : %d" %(x_dim, y_dim))
            print("Maximum label : ", y_depth, "\n")
        return (num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth)
    else:
        assert (len(train_data) == len(valid_data) and len(train_data) == len(test_data)), "Different number of groups in train/validation data"
        num_group = len(train_data)

        bool_num_task = [(len(train_data[0]) == len(train_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in train data"
        bool_num_task = [(len(valid_data[0]) == len(valid_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in validation data"
        assert (len(train_data[0])==len(valid_data[0]) and len(train_data[0])==len(test_data[0])), "Different number of tasks in train/validation/test data"
        num_task = len(train_data[0])

        num_train, num_valid, num_test = [train_data[0][x][0].shape[0] for x in range(num_task)], [valid_data[0][x][0].shape[0] for x in range(num_task)], [test_data[0][x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0][0].shape[1], 0
        y_depth = [int(np.amax(x[1])+1) for x in train_data[0]]
        if print_info:
            print("Tasks : %d, Groups of training/valid : %d\n" %(num_task, num_group))
            print("Train data : ", num_train, ", Validation data : ", num_valid, ", Test data : ", num_test)
            print("Input dim : %d, Label dim : %d" %(x_dim, y_dim))
            print("Maximum label : ", y_depth, "\n")
        return (num_task, num_group, num_train, num_valid, num_test, x_dim, y_dim, y_depth)


# couch(13), table(58), paper_clip(41), clipboards(11), bucket(6), notebook(38), bed(3), trash_can(62)
_officehomeincr_task_labels = [[13, 58, 41],
                               [13, 58, 41, 11],
                               [13, 58, 41, 11, 6],
                               [13, 58, 41, 11, 6, 38],
                               [13, 58, 41, 11, 6, 38, 3],
                               [13, 58, 41, 11, 6, 38, 3, 62]]


def officehome_incremental_data(data_file_name, num_train_ratio, num_valid_ratio, num_test_ratio, num_train_group, img_size, save_as_mat=False):
    assert (num_train_ratio + num_valid_ratio + num_test_ratio <= 1.0), "Sum of the given ratio of data should be less than or equal to 1"

    curr_path = os.getcwd()
    if not ('Data' in os.listdir(curr_path)):
        os.mkdir('./Data')

    data_path = curr_path + '/Data'
    if data_file_name in os.listdir(data_path):
        with open('./Data/' + data_file_name, 'rb') as fobj:
            if sys.version_info.major < 3:
                train_data, validation_data, test_data = pickle.load(fobj)
            else:
                train_data, validation_data, test_data = pickle.load(fobj, encoding='latin1')
            print('Successfully load data')
    else:
        data_types = ['Product']
        raw_images = read_officehome_all_images(data_types, img_size)

        train_data, validation_data, test_data = [], [], []
        for group_cnt in range(num_train_group):
            train_data_tmp, validation_data_tmp, test_data_tmp = [], [], []

            for data_type in data_types:
                for task_cnt, (task_list) in enumerate(_officehomeincr_task_labels):
                    for class_cnt, (class_label) in enumerate(task_list):
                        imgs_in_class_subtask = raw_images[data_type][class_label]
                        num_imgs = imgs_in_class_subtask.shape[0]
                        num_train, num_valid, num_test = int(num_imgs*num_train_ratio), int(num_imgs*num_valid_ratio), int(num_imgs*num_test_ratio)
                        img_indices = list(range(num_imgs))
                        shuffle(img_indices)

                        train_x_tmp, train_y_tmp = imgs_in_class_subtask[img_indices[0:num_train],:], np.ones(num_train, dtype=np.int32)*class_cnt
                        valid_x_tmp, valid_y_tmp = imgs_in_class_subtask[img_indices[num_train:num_train+num_valid],:], np.ones(num_valid, dtype=np.int32)*class_cnt
                        test_x_tmp, test_y_tmp = imgs_in_class_subtask[img_indices[num_train+num_valid:num_train+num_valid+num_test],:], np.ones(num_test, dtype=np.int32)*class_cnt

                        if class_cnt < 1:
                            train_x, train_y = train_x_tmp, train_y_tmp
                            valid_x, valid_y = valid_x_tmp, valid_y_tmp
                            test_x, test_y = test_x_tmp, test_y_tmp
                        else:
                            train_x, train_y = np.concatenate((train_x, train_x_tmp), axis=0), np.concatenate((train_y, train_y_tmp), axis=0)
                            valid_x, valid_y = np.concatenate((valid_x, valid_x_tmp), axis=0), np.concatenate((valid_y, valid_y_tmp), axis=0)
                            test_x, test_y = np.concatenate((test_x, test_x_tmp), axis=0), np.concatenate((test_y, test_y_tmp), axis=0)

                    train_x, train_y = shuffle_data_x_and_y(train_x, train_y)
                    valid_x, valid_y = shuffle_data_x_and_y(valid_x, valid_y)
                    test_x, test_y = shuffle_data_x_and_y(test_x, test_y)

                    train_data_tmp.append( ( np.array(train_x), np.array(train_y) ) )
                    validation_data_tmp.append( ( np.array(valid_x), np.array(valid_y) ) )
                    test_data_tmp.append( ( np.array(test_x), np.array(test_y) ) )

            train_data.append(train_data_tmp)
            validation_data.append(validation_data_tmp)
            test_data.append(test_data_tmp)
            num_tasks = len(train_data_tmp)

        #### save data
        if save_as_mat:
            data_to_save_in_mat = _data_save_to_mat(train_data, validation_data, test_data, num_train_group, num_tasks)
            savemat('./Data/'+data_file_name[0:-4]+'.mat', data_to_save_in_mat)

        with open('./Data/' + data_file_name, 'wb') as fobj:
            pickle.dump([train_data, validation_data, test_data], fobj)
            print('Successfully generate/save data')
    return (train_data, validation_data, test_data)

def officehome_incremental_data_print_info(train_data, valid_data, test_data, no_group=False, print_info=True):
    if no_group:
        assert (len(train_data) == len(valid_data) and len(train_data) == len(test_data)), "Different number of tasks in train/validation data"
        num_task = len(train_data)

        num_train, num_valid, num_test = [train_data[x][0].shape[0] for x in range(num_task)], [valid_data[x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0].shape[1], 0
        y_depth = [int(np.amax(x[1])+1) for x in train_data]
        if print_info:
            print("Train data : ", num_train, ", Validation data : ", num_valid, ", Test data : ", num_test)
            print("Input dim : %d" %(x_dim))
            print("Output dim : ", y_depth, "\n")
        return (num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth)
    else:
        assert (len(train_data) == len(valid_data) and len(train_data) == len(test_data)), "Different number of groups in train/validation data"
        num_group = len(train_data)

        bool_num_task = [(len(train_data[0]) == len(train_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in train data"
        bool_num_task = [(len(valid_data[0]) == len(valid_data[x])) for x in range(1, num_group)]
        assert all(bool_num_task), "Different number of tasks in some of groups in validation data"
        assert (len(train_data[0])==len(valid_data[0]) and len(train_data[0])==len(test_data[0])), "Different number of tasks in train/validation/test data"
        num_task = len(train_data[0])

        num_train, num_valid, num_test = [train_data[0][x][0].shape[0] for x in range(num_task)], [valid_data[0][x][0].shape[0] for x in range(num_task)], [test_data[0][x][0].shape[0] for x in range(num_task)]
        x_dim, y_dim = train_data[0][0][0].shape[1], 0
        y_depth = [int(np.amax(x[1])+1) for x in train_data[0]]
        if print_info:
            print("Tasks : %d, Groups of training/valid : %d\n" %(num_task, num_group))
            print("Train data : ", num_train, ", Validation data : ", num_valid, ", Test data : ", num_test)
            print("Input dim : %d" %(x_dim))
            print("Output dim : ", y_depth, "\n")
        return (num_task, num_group, num_train, num_valid, num_test, x_dim, y_dim, y_depth)


def each_experiment_info_reader(task_file_name, file_path):
    if not (task_file_name in os.listdir(file_path)):
        raise ValueError("Given name of file doesn't exist in the path!")
    experiment_info = []
    with open(file_path+'/'+task_file_name, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        for row in datareader:
            if len(row) > 0:
                if 'Task' in row[0]:
                    if ('task_info' in globals() or 'task_info' in locals()):
                        task_info['TrainIndices'] = np.array(task_info_train_indices_tmp)
                        task_info['ValidationIndices'] = np.array(task_info_valid_indices_tmp)
                        experiment_info.append(task_info)
                        del task_info, task_info_train_indices_tmp, task_info_valid_indices_tmp
                    task_info = {}
                    task_info['Task'] = int(row[1])
                    mode=0
                elif 'Classes' in row[0]:
                    task_info['Classes'] = [int(a) for a in row[1:]]
                    mode=0
                elif 'Train' in row[0]:
                    task_info_train_indices_tmp, mode = [], 1
                elif 'Valid' in row[0]:
                    task_info_valid_indices_tmp, mode = [], 2
                elif 'Noise' in row[0]:
                    ## Mean and variance of gaussian noise
                    task_info['Noise'] = [float(a) for a in row[1:]]
                elif 'ChannelSwap' in row[0]:
                    ## Swap input channels for varying input-level task-similarity (only for dataset with channels>1)
                    task_info['ChannelSwap'] = [int(a) for a in row[1:]]
                else:
                    assert (len(row)>1), "Class-index pair is in wrong format!"
                    assert (int(row[0]) in task_info['Classes']), "Wrong image class"
                    if mode == 1:
                        task_info_train_indices_tmp.append([int(row[0]), int(row[1])])
                    elif mode == 2:
                        task_info_valid_indices_tmp.append([int(row[0]), int(row[1])])
        if ('task_info' in globals() or 'task_info' in locals()):
            if ('task_info_train_indices_tmp' in globals() or 'task_info_train_indices_tmp' in locals()) and ('task_info_valid_indices_tmp' in globals() or 'task_info_valid_indices_tmp' in locals()):
                if len(task_info_train_indices_tmp) > 0 and len(task_info_valid_indices_tmp) > 0:
                    task_info['TrainIndices'] = np.array(task_info_train_indices_tmp)
                    task_info['ValidationIndices'] = np.array(task_info_valid_indices_tmp)
                    experiment_info.append(task_info)
    return experiment_info

def experiment_info_reader(exp_file_name, file_path, num_seeds):
    experiments_design = []
    for exp_cnt in range(num_seeds):
        actual_file_name = exp_file_name+'_s'+str(exp_cnt)+'.csv'
        if not (actual_file_name in os.listdir(file_path)):
            raise ValueError("Given name of file doesn't exist in the path!")
        experiment_info = each_experiment_info_reader(actual_file_name, file_path)
        experiments_design.append(experiment_info)
    return experiments_design

#### write experiment design into csv file
def experiment_info_writer(exp_file_name, file_path, experiments_design):
    for exp_cnt, (experiment_info) in enumerate(experiments_design):
        actual_file_name = exp_file_name+'_s'+str(exp_cnt)+'.csv'
        if (actual_file_name in os.listdir(file_path)):
            print("\n\nExperiment design with same name already exists!!")
            decision = input("\tDo overwrite?\t")
            if not (decision.lower() == 'y' or decision.lower() == 'yes'):
                raise ValueError("Given name of file cannot be created in the path!")

        with open(file_path+'/'+actual_file_name, 'w', newline='') as csvfile:
            datawriter = csv.writer(csvfile, delimiter=',')
            for task_info in experiment_info:
                datawriter.writerow(['Task', task_info['Task']])
                datawriter.writerow(['Classes']+task_info['Classes'])
                if 'Noise' in task_info.keys():
                    ## Mean and variance of gaussian noise
                    datawriter.writerow(['Noise']+task_info['Noise'])
                if 'ChannelSwap' in task_info.keys():
                    ## Swap input channels for varying input-level task-similarity (only for dataset with channels>1)
                    datawriter.writerow(['ChannelSwap']+task_info['ChannelSwap'])
                datawriter.writerow(['TrainIndices'])
                for data_cnt in range(task_info['TrainIndices'].shape[0]):
                    datawriter.writerow([task_info['TrainIndices'][data_cnt, 0], task_info['TrainIndices'][data_cnt, 1]])
                datawriter.writerow(['ValidationIndices'])
                for data_cnt in range(task_info['ValidationIndices'].shape[0]):
                    datawriter.writerow([task_info['ValidationIndices'][data_cnt, 0], task_info['ValidationIndices'][data_cnt, 1]])


#### make pairs of classes for tasks and lists of data indices for training and validation set
def experiment_designer(data_percent, num_tasks, num_seeds, valid_data_ratio_to_whole, num_data_per_class, num_classes_per_task, allowNoise=False, allowChannelSwap=False):
    experiments_design = []
    num_usable_data_per_class, num_classes = [int(a*data_percent) for a in num_data_per_class], len(num_data_per_class)
    classes_in_tasks, task_ids = [], []
    noises_in_tasks, channel_swap_in_tasks = [], []
    for task_cnt in range(num_tasks):
        classes_for_this_task = []
        while len(classes_for_this_task) < num_classes_per_task:
            a = randint(0, num_classes-1)
            if len(classes_for_this_task) < 1:
                classes_for_this_task.append(a)
            elif not any([b == a for b in classes_for_this_task]):
                classes_for_this_task.append(a)

            if len(classes_for_this_task) == num_classes_per_task:
                classes_copy = list(classes_for_this_task)
                classes_copy.sort()
                task_id = sum([a*(10**i) for i, (a) in enumerate(classes_copy)])
                if task_id in task_ids:
                    classes_for_this_task = []
                else:
                    classes_in_tasks.append(classes_for_this_task)
                    task_ids.append(task_id)
                    if allowNoise:
                        mean, std = np.random.randint(-2, 2), np.random.randint(0, 2)
                        if mean == 0 and std == 0:
                            noises_in_tasks.append(None)
                        else:
                            noises_in_tasks.append([0.025*mean, 0.025*std])
                    else:
                        noises_in_tasks.append(None)

                    if allowChannelSwap:
                        new_channel_list = [0, 1, 2]
                        shuffle(new_channel_list)
                        if all([tmp_a == tmp_b for (tmp_a, tmp_b) in zip(new_channel_list, [0, 1, 2])]):
                            channel_swap_in_tasks.append(None)
                        else:
                            channel_swap_in_tasks.append(new_channel_list)
                    else:
                        channel_swap_in_tasks.append(None)

    for seed_cnt in range(num_seeds):
        experiment_info = []
        for task_cnt, (classes_in_this_task, noise_in_this_task, channel_swap_in_this_task) in enumerate(zip(classes_in_tasks, noises_in_tasks, channel_swap_in_tasks)):
            task_info, temp_train_indices, temp_valid_indices = {}, [], []
            task_info['Task'] = task_cnt
            task_info['Classes'] = classes_in_this_task
            if noise_in_this_task is not None:
                task_info['Noise'] = list(noise_in_this_task)
            if channel_swap_in_this_task is not None:
                task_info['ChannelSwap'] = list(channel_swap_in_this_task)

            for spec_class in classes_in_this_task:
                data_indices = list(range(num_data_per_class[spec_class]))
                shuffle(data_indices)
                for index_cnt in range(num_usable_data_per_class[spec_class]):
                    if index_cnt < valid_data_ratio_to_whole*num_usable_data_per_class[spec_class]:
                        temp_valid_indices.append([spec_class, data_indices[index_cnt]])
                    else:
                        temp_train_indices.append([spec_class, data_indices[index_cnt]])
            task_info['TrainIndices'] = np.array(temp_train_indices)
            task_info['ValidationIndices'] = np.array(temp_valid_indices)
            experiment_info.append(task_info)
        experiments_design.append(experiment_info)
    return experiments_design


def print_experiment_design(exp_design, data_type, print_info=True):
    if 'fashion' in data_type.lower():
        x_dim, y_dim = 28*28, 0
    elif 'stl' in data_type.lower() or 'stl10' in data_type.lower():
        x_dim, y_dim = 96*96*3, 0
    else:
        x_dim, y_dim = None, None

    if type(exp_design) == list:
        if type(exp_design[0]) == dict:
            # exp design containing a series of sub-tasks for one random seed
            num_task = len(exp_design)
            num_train, num_valid = [exp_design[x]['TrainIndices'].shape[0] for x in range(num_task)], [exp_design[x]['ValidationIndices'].shape[0] for x in range(num_task)]
            y_depth = [len(exp_design[x]['Classes']) for x in range(num_task)]

            if print_info:
                print("Train data : ", num_train, ", Validation data : ", num_valid)
                print("Input dim : %d, Label dim : %d" %(x_dim, y_dim))
                print("Maximum label : ", y_depth, "\n")
            return (num_task, num_train, num_valid, x_dim, y_dim, y_depth)
        elif type(exp_design[0]) == list:
            # a list of exp designs (different random seeds)
            num_group, num_task = len(exp_design), len(exp_design[0])
            assert all([(len(exp_design[0]) == len(exp_design[x])) for x in range(1, num_group)]), "Different number of tasks in some of groups in train data"

            num_train, num_valid = [exp_design[0][x]['TrainIndices'].shape[0] for x in range(num_task)], [exp_design[0][x]['ValidationIndices'].shape[0] for x in range(num_task)]
            y_depth = [len(exp_design[0][x]['Classes']) for x in range(num_task)]

            if print_info:
                print("Tasks : %d, Groups of training/valid : %d\n" %(num_task, num_group))
                print("Train data : ", num_train, ", Validation data : ", num_valid)
                print("Input dim : %d, Label dim : %d" %(x_dim, y_dim))
                print("Maximum label : ", y_depth, "\n")
            return (num_task, num_group, num_train, num_valid, x_dim, y_dim, y_depth)
        else:
            raise ValueError("Given experiment design is in wrong format!")
    else:
        raise ValueError("Given experiment design is in wrong format!")

def print_data_info(train_data, valid_data, test_data, print_info=False):
    num_task = len(train_data)

    num_train, num_valid, num_test = [train_data[x][0].shape[0] for x in range(num_task)], [valid_data[x][0].shape[0] for x in range(num_task)], [test_data[x][0].shape[0] for x in range(num_task)]
    x_dim, y_dim = train_data[0][0].shape[1], 0
    y_depth = [int(np.amax(x[1])+1) for x in train_data]
    if print_info:
        print("Tasks : %d\nTrain data : %d, Validation data : %d, Test data : %d" %(num_task, num_train, num_valid, num_test))
        print("Input dim : %d, Label dim : %d" %(x_dim, y_dim))
        print("Maximum label : ", y_depth, "\n")
    return (num_task, num_train, num_valid, num_test, x_dim, y_dim, y_depth)


def data_handler_for_experiment(categorized_train_data, categorized_test_data, experiment_design, img_shape=None):
    ## Caution!
    ## "experiment design" is a list of sub-tasks, NOT a list of L2M experiments (each L2M experiment == a list of sub-tasks)

    ## each of train/validation/test data format:
    ## data[task_index][x or y][instance_index] <- list of tuples of np 2D(x)/1D(y) array
    train_data, validation_data, test_data = [], [], []
    for task_design in experiment_design:
        train_x_this_task, train_y_this_task = [], []
        valid_x_this_task, valid_y_this_task = [], []
        test_x_this_task, test_y_this_task = [], []

        if 'Noise' in task_design.keys():
            add_noise, noise_info = True, task_design['Noise']
        else:
            add_noise, noise_info = False, None

        if 'ChannelSwap' in task_design.keys():
            swap_channel, channel_swap_info = True, task_design['ChannelSwap']
            assert (type(img_shape)==list), "Must give the shape of image to swap channels!"
        else:
            swap_channel, channel_swap_info = False, None

        classes_for_this_task = task_design['Classes']
        for class_cnt, (class_label) in enumerate(classes_for_this_task):
            ## training data
            indices_of_indices = np.nonzero(task_design['TrainIndices'][:,0]==class_label)
            num_data = len(indices_of_indices[0])
            if num_data < 1:
                raise ValueError("Experiment design error: class description mismatch within one task!")
            train_data_indices = task_design['TrainIndices'][indices_of_indices, 1]
            train_x_this_task.append(np.array(categorized_train_data[class_label][train_data_indices,:][0]))
            train_y_this_task.append(class_cnt*np.ones([num_data, 1], dtype=np.int32))

            ## validation data
            indices_of_indices = np.nonzero(task_design['ValidationIndices'][:,0]==class_label)
            num_data = len(indices_of_indices[0])
            if num_data < 1:
                raise ValueError("Experiment design error: class description mismatch within one task!")
            valid_data_indices = task_design['ValidationIndices'][indices_of_indices, 1]
            valid_x_this_task.append(np.array(categorized_train_data[class_label][valid_data_indices,:][0]))
            valid_y_this_task.append(class_cnt*np.ones([num_data, 1], dtype=np.int32))

            ## test data
            test_x_this_task.append(np.array(categorized_test_data[class_label]))
            test_y_this_task.append(class_cnt*np.ones([test_x_this_task[-1].shape[0], 1], dtype=np.int32))

        train_x_this_task, train_y_this_task = np.concatenate(train_x_this_task, axis=0), np.concatenate(train_y_this_task, axis=0).reshape(-1)
        valid_x_this_task, valid_y_this_task = np.concatenate(valid_x_this_task, axis=0), np.concatenate(valid_y_this_task, axis=0).reshape(-1)
        test_x_this_task, test_y_this_task = np.concatenate(test_x_this_task, axis=0), np.concatenate(test_y_this_task, axis=0).reshape(-1)

        if add_noise:
            train_x_this_task = np.clip(train_x_this_task + np.random.normal(loc=noise_info[0], scale=noise_info[1], size=train_x_this_task.shape), -0.5, 0.5)
            valid_x_this_task = np.clip(valid_x_this_task + np.random.normal(loc=noise_info[0], scale=noise_info[1], size=valid_x_this_task.shape), -0.5, 0.5)
            test_x_this_task = np.clip(test_x_this_task + np.random.normal(loc=noise_info[0], scale=noise_info[1], size=test_x_this_task.shape), -0.5, 0.5)

        if swap_channel:
            num_train, num_valid, num_test = train_x_this_task.shape[0], valid_x_this_task.shape[0], test_x_this_task.shape[0]
            tmp_train_x, tmp_valid_x, tmp_test_x = train_x_this_task.reshape([num_train]+img_shape), valid_x_this_task.reshape([num_valid]+img_shape), test_x_this_task.reshape([num_test]+img_shape)

            train_x_this_task = tmp_train_x[:,:,:,channel_swap_info].reshape([num_train, -1])
            valid_x_this_task = tmp_valid_x[:,:,:,channel_swap_info].reshape([num_valid, -1])
            test_x_this_task = tmp_test_x[:,:,:,channel_swap_info].reshape([num_test, -1])

        train_data.append( (train_x_this_task, train_y_this_task) )
        validation_data.append( (valid_x_this_task, valid_y_this_task) )
        test_data.append( (test_x_this_task, test_y_this_task) )
    return train_data, validation_data, test_data



################################### STL-10 data
#### read/generate csv file for the information about experiment (classes and indices of images in each sub-task)
def read_raw_stl10(data_path):
    with open(data_path+'/raw_data/train_X.bin', 'rb') as fobj:
        train_data_x_tmp = np.fromfile(fobj, dtype=np.uint8)
        train_data_x = np.transpose(np.reshape(train_data_x_tmp, [-1, 3, 96, 96]), (0, 3, 2, 1)).astype(dtype=np.float32)/255.0 - 0.5
        train_data_x = train_data_x.reshape([-1, 96*96*3])

    with open(data_path+'/raw_data/train_y.bin', 'rb') as fobj:
        train_data_y_tmp = np.fromfile(fobj, dtype=np.uint8)
        train_data_y = train_data_y_tmp - 1

    with open(data_path+'/raw_data/test_X.bin', 'rb') as fobj:
        test_data_x_tmp = np.fromfile(fobj, dtype=np.uint8)
        test_data_x = np.transpose(np.reshape(test_data_x_tmp, [-1, 3, 96, 96]), (0, 3, 2, 1)).astype(dtype=np.float32)/255.0 - 0.5
        test_data_x = test_data_x.reshape([-1, 96*96*3])

    with open(data_path+'/raw_data/test_y.bin', 'rb') as fobj:
        test_data_y_tmp = np.fromfile(fobj, dtype=np.uint8)
        test_data_y = test_data_y_tmp - 1
    ## x : 2D matrix of float32 images (NHWC) [-0.5, 0.5], y : 1D array of uint8 labels
    return (train_data_x, train_data_y, test_data_x, test_data_y)

def stl10_data(experiment_file_base_name, valid_data_ratio_to_whole, num_train_group, num_tasks=5, data_percent=1.0, num_classes_per_task=2, allowNoise=False, allowChannelSwap=False):
    curr_path = os.getcwd()
    if not ('Data' in os.listdir(curr_path)):
        os.mkdir('./Data')

    data_path = curr_path + '/Data/STL-10'
    raw_train_images, raw_train_labels, raw_test_images, raw_test_labels = read_raw_stl10(data_path)

    ### make lists of data for each class
    temp_categorized_train_data = data_class_split((raw_train_images, raw_train_labels), 10)
    temp_categorized_test_data = data_class_split((raw_test_images, raw_test_labels), 10)

    categorized_train_data = [np.array(A) for A in temp_categorized_train_data]
    categorized_test_data = [np.array(A) for A in temp_categorized_test_data]
    del raw_train_images, raw_train_labels, raw_test_images, raw_test_labels, temp_categorized_train_data, temp_categorized_test_data
    num_train_data_each_class = [A.shape[0] for A in categorized_train_data]

    try:
        ### make code to read list of data indices for sub-tasks
        print("\n\nRead the set of experiments using STL-10 data!")
        experiments_design = experiment_info_reader(experiment_file_base_name, data_path, num_train_group)
        print("\tSuccessfully read experiment set-up using STL-10 data!\n\n")
    except:
        ### design sub-tasks setting
        print("\n\tNo experiment design exists!\nGenerate the set of experiments using STL-10 data!")
        experiments_design = experiment_designer(data_percent, num_tasks, num_train_group, valid_data_ratio_to_whole, num_train_data_each_class, num_classes_per_task=num_classes_per_task, allowNoise=allowNoise, allowChannelSwap=allowChannelSwap)

        ### make code to save list of data indices for sub-tasks into file
        experiment_info_writer(experiment_file_base_name, data_path, experiments_design)
        print("\tSuccessfully generated and saved experiment set-up using STL-10 data!\n\n")
    return categorized_train_data, categorized_test_data, experiments_design



#################################### Miscellaneous

#### save data in mat file
def _data_save_to_mat(train_data, validation_data, test_data, num_train_group, num_task):
    train_data_cell_x = np.zeros((num_train_group,), dtype=np.object)
    train_data_cell_y = np.zeros((num_train_group,), dtype=np.object)
    valid_data_cell_x = np.zeros((num_train_group,), dtype=np.object)
    valid_data_cell_y = np.zeros((num_train_group,), dtype=np.object)
    test_data_cell_x = np.zeros((num_task,), dtype=np.object)
    test_data_cell_y = np.zeros((num_task,), dtype=np.object)

    for group_cnt in range(num_train_group):
        train_tmp_x, valid_tmp_x = np.zeros((num_task,), dtype=np.object), np.zeros((num_task,), dtype=np.object)
        train_tmp_y, valid_tmp_y = np.zeros((num_task,), dtype=np.object), np.zeros((num_task,), dtype=np.object)
        for task_cnt in range(num_task):
            train_tmp_x[task_cnt] = train_data[group_cnt][task_cnt][0]
            train_tmp_y[task_cnt] = train_data[group_cnt][task_cnt][1]
            valid_tmp_x[task_cnt] = validation_data[group_cnt][task_cnt][0]
            valid_tmp_y[task_cnt] = validation_data[group_cnt][task_cnt][1]
        train_data_cell_x[group_cnt] = train_tmp_x
        train_data_cell_y[group_cnt] = train_tmp_y
        valid_data_cell_x[group_cnt] = valid_tmp_x
        valid_data_cell_y[group_cnt] = valid_tmp_y

    for task_cnt in range(num_task):
        test_data_cell_x[task_cnt] = test_data[task_cnt][0]
        test_data_cell_y[task_cnt] = test_data[task_cnt][1]

    data_save_struct = {}
    data_save_struct['train_data_x'] = train_data_cell_x
    data_save_struct['train_data_y'] = train_data_cell_y
    data_save_struct['valid_data_x'] = valid_data_cell_x
    data_save_struct['valid_data_y'] = valid_data_cell_y
    data_save_struct['test_data_x'] = test_data_cell_x
    data_save_struct['test_data_y'] = test_data_cell_y
    return data_save_struct


if __name__ == '__main__':
    print('Nothing to do here!')
