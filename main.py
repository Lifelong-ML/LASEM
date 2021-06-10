from os import getcwd, listdir, mkdir
from utils.utils_env_cl import num_data_points, model_setup
from classification.gen_data import mnist_data, mnist_data_print_info, cifar10_data, cifar100_data, cifar_data_print_info, officehome_data, officehome_data_print_info
from classification.gen_data import stl10_data, print_experiment_design ## New format of experiment design
from classification.train_wrapper import train_run_for_each_model, train_run_for_each_model_v2

def main():
    import argparse
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--gpu', help='GPU device ID', type=int, default=-1)
    parser.add_argument('--data_type', help='Type of Data (MNIST5/MNIST10/CIFAR10/CIFAR100)', type=str, default='MNIST5')
    parser.add_argument('--data_percent', help='Percentage of train data to be used', type=int, default=100)
    parser.add_argument('--model_type', help='Architecture of Model(STL/SNN/HPS/TF/PROG/Deconv/DeconvTM/DeconvTM2/DeconvTM_rev/DeconvTM2_rev/DeconvTM2_reshape)', type=str, default='STL')
    parser.add_argument('--save_mat_name', help='Name of file to save training results', type=str, default='delete_this.mat')
    parser.add_argument('--test_type', help='For hyper-parameter search', type=int, default=0)
    parser.add_argument('--task_order_type', help='Choose the sequence of tasks presented to LL model', type=int, default=0)
    parser.add_argument('--cnn_padtype_valid', help='Set CNN padding type VALID', action='store_false', default=True)
    parser.add_argument('--lifelong', help='Train in lifelong learning setting', action='store_true', default=False)
    parser.add_argument('--saveparam', help='Save parameter of NN', action='store_true', default=False)
    parser.add_argument('--savegraph', help='Save graph of NN', action='store_true', default=False)
    parser.add_argument('--tensorfactor_param_path', help='Path to parameters initializing tensor factorized model(below Result, above run0/run1/etc', type=str, default=None)
    parser.add_argument('--num_classes', help='Number of classes for each sub-task', type=int, default=2)

    parser.add_argument('--skip_connect_test_type', help='For testing several ways to make skip connections', type=int, default=0)
    parser.add_argument('--highway_connect_test_type', help='For testing several ways to make skip connections (highway net)', type=int, default=0)

    parser.add_argument('--num_clayers', help='Number of conv layers for Office-Home experiment', type=int, default=-1)
    parser.add_argument('--phase1_max_epoch', help='Number of epochs in training phase 1 of Hybrid DF-CNN auto sharing', type=int, default=100)

    parser.add_argument('--reset_prior', help='Reset prior of configs each epoch', action='store_true', default=False)
    parser.add_argument('--em_fix_maxC', help='EM analysis - configuration of max assigned probability', type=str, default='top1')
    parser.add_argument('--em_maxP', help='EM analysis - probability of the config with max assigned probability', type=float, default=0.9)
    parser.add_argument('--em_cnt_prior', help='EM method - prior probability is based on the count of mini-batch', action='store_true', default=False)
    parser.add_argument('--em_prior_init_cnt', help='EM method - initial cnt for count-based prior probability', type=float, default=1)

    parser.add_argument('--darts_approx_order', help='Order of approximation of DARTS', type=int, default=1)
    parser.add_argument('--data_augment', help='Do data augmentation in mini-batch', action='store_true', default=False)

    ## arguments to train STL model with more data (only for STL model)
    parser.add_argument('--stl_analysis', help='Run STL analysis by providing more training data', action='store_true', default=False)
    parser.add_argument('--stl_task_to_learn', help='(For STL analysis) task to learn', type=int, default=-1)
    parser.add_argument('--stl_total_data_ratio', help='Ratio of the amount of data to the amount of data for MTL training from task 1 to T (stl_task_to_learn). 1.0 should be max', type=float, default=1.0)

    ## arguments to compute mutual information of flexible DF-CNN (or Hybrid DF-CNN in paper)
    parser.add_argument('--mutual_info', help='Compute mutual information of each layer after training', action='store_true', default=False)
    parser.add_argument('--mutual_info_alpha', help='Value of alpha for Renyi alpha entropy', type=float, default=1.01)
    parser.add_argument('--mutual_info_kernel_h', help='Value of h for RBF kernel', type=float, default=1.0)
    parser.add_argument('--mutual_info_kernel_h_backward', help='Value of h for RBF kernel', type=float, default=1.0)


    args = parser.parse_args()

    gpu_device_num = args.gpu
    if gpu_device_num > -1:
        use_gpu = True
    else:
        use_gpu = False
    do_lifelong = args.lifelong

    if not 'Result' in listdir(getcwd()):
        mkdir('Result')

    mat_file_name = args.save_mat_name

    data_type, data_percent = args.data_type.lower(), args.data_percent
    data_hyperpara = {}
    data_hyperpara['num_train_group'] = 5
    data_hyperpara['multi_class_label'] = False

    train_hyperpara = {}
    train_hyperpara['improvement_threshold'] = 1.002      # for accuracy (maximizing it)
    train_hyperpara['patience_multiplier'] = 1.5

    train_hyperpara['stl_analysis'] = args.stl_analysis
    train_hyperpara['LEEP_score'] = False
    if args.stl_analysis:
        assert (args.model_type.lower()=='stl' and args.stl_task_to_learn>=0), "Some arguments don't satisfy requirement of STL analysis."
        train_hyperpara['stl_task_to_learn'] = args.stl_task_to_learn
        train_hyperpara['stl_total_data_ratio'] = max(min(args.stl_total_data_ratio, 1.0), 0.0)

    train_hyperpara['mutual_info'] = args.mutual_info
    train_hyperpara['mutual_info_alpha'] = args.mutual_info_alpha
    train_hyperpara['mutual_info_kernel_h'] = args.mutual_info_kernel_h
    train_hyperpara['mutual_info_kernel_h_backward'] = args.mutual_info_kernel_h_backward

    train_hyperpara['em_cnt_prior'] = args.em_cnt_prior
    train_hyperpara['em_prior_init_cnt'] = args.em_prior_init_cnt

    train_hyperpara['data_augment'] = args.data_augment
    if train_hyperpara['data_augment']:
        print("\nData Augmentation will be applied during training!\n")

    if 'mnist' in data_type and 'fashion' not in data_type:
        data_hyperpara['image_dimension'] = [28, 28, 1]
        data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'] = num_data_points(data_type, data_percent)
        if '5' in data_type:
            data_hyperpara['num_tasks'] = 5
        elif '10' in data_type:
            data_hyperpara['num_tasks'] = 10
        data_file_name = 'mnist_mtl_data_group_' + str(data_hyperpara['num_train_data']) + '_' + str(data_hyperpara['num_valid_data']) + '_' + str(data_hyperpara['num_test_data']) + '_' + str(data_hyperpara['num_tasks']) + '.pkl'

        train_hyperpara['num_run_per_model'] = 5
        train_hyperpara['train_valid_data_group'] = list(range(5)) + list(range(5))
        train_hyperpara['lr'] = 0.001
        train_hyperpara['lr_decay'] = 1.0/250.0
        train_hyperpara['learning_step_max'] = 500
        train_hyperpara['patience'] = 500
        train_hyperpara['task_order'] = list(range(data_hyperpara['num_tasks']))

        train_data, validation_data, test_data = mnist_data(data_file_name, data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'], data_hyperpara['num_train_group'], data_hyperpara['num_tasks'], data_percent)
        mnist_data_print_info(train_data, validation_data, test_data)
        classification_prob=True
    elif ('cifar10' in data_type) and not ('cifar100' in data_type):
        data_hyperpara['image_dimension'] = [32, 32, 3]
        data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'] = num_data_points(data_type, data_percent)
        if '_5' in data_type:
            data_hyperpara['num_tasks'] = 5
        elif '_10' in data_type:
            data_hyperpara['num_tasks'] = 10
        data_file_name = 'cifar10_mtl_data_group_' + str(data_hyperpara['num_train_data']) + '_' + str(data_hyperpara['num_valid_data']) + '_' + str(data_hyperpara['num_test_data']) + '_' + str(data_hyperpara['num_tasks']) + '.pkl'

        train_hyperpara['num_run_per_model'] = 5
        train_hyperpara['train_valid_data_group'] = range(5)
        train_hyperpara['lr'] = 0.00025
        train_hyperpara['lr_decay'] = 1.0/1000.0
        train_hyperpara['learning_step_max'] = 2000
        train_hyperpara['patience'] = 2000
        train_hyperpara['task_order'] = list(range(data_hyperpara['num_tasks']))

        train_data, validation_data, test_data = cifar10_data(data_file_name, data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'], data_hyperpara['num_train_group'], data_hyperpara['num_tasks'], multiclass=data_hyperpara['multi_class_label'], data_percent=data_percent)
        cifar_data_print_info(train_data, validation_data, test_data)
        classification_prob=True
    elif 'cifar100' in data_type:
        data_hyperpara['multi_class_label'] = True
        data_hyperpara['image_dimension'] = [32, 32, 3]
        data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'] = num_data_points(data_type, data_percent)
        if '_10' in data_type:
            data_hyperpara['num_tasks'] = 10
        elif '_20' in data_type:
            data_hyperpara['num_tasks'] = 20
        elif '_40' in data_type:
            data_hyperpara['num_tasks'] = 40
        data_file_name = 'cifar100_mtl_data_group_' + str(data_hyperpara['num_train_data']) + '_' + str(data_hyperpara['num_valid_data']) + '_' + str(data_hyperpara['num_test_data']) + '_' + str(data_hyperpara['num_tasks']) + '.pkl'

        train_hyperpara['num_run_per_model'] = 5
        train_hyperpara['train_valid_data_group'] = range(5)
        train_hyperpara['lr'] = 0.0001
        train_hyperpara['lr_decay'] = 1.0/4000.0
        train_hyperpara['patience'] = 2000
        train_hyperpara['task_order'] = list(range(data_hyperpara['num_tasks']))
        if args.task_order_type == 1:
            train_hyperpara['task_order'] = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif args.task_order_type == 2:
            train_hyperpara['task_order'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


        train_hyperpara['patience'] = 5
        train_hyperpara['num_run_per_model'] = 2


        train_hyperpara['learning_step_max'] = len(train_hyperpara['task_order']) * train_hyperpara['patience']

        train_data, validation_data, test_data = cifar100_data(data_file_name, data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'], data_hyperpara['num_train_group'], data_hyperpara['num_tasks'], multiclass=data_hyperpara['multi_class_label'])
        cifar_data_print_info(train_data, validation_data, test_data)
        classification_prob=True

    elif 'officehome' in data_type:
        data_hyperpara['multi_class_label'] = True
        data_hyperpara['image_dimension'] = [128, 128, 3]
        data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'] = 0.6, 0.1, 0.3
        data_hyperpara['num_classes'] = 13
        data_hyperpara['num_tasks'] = 10

        data_file_name = 'officehome_mtl_data_group_' + str(data_hyperpara['num_train_data']) + '_' + str(data_hyperpara['num_valid_data']) + '_' + str(data_hyperpara['num_test_data']) + '_t' + str(data_hyperpara['num_tasks']) + '_c' + str(data_hyperpara['num_classes']) + '_i' + str(data_hyperpara['image_dimension'][0]) + '.pkl'

        train_hyperpara['num_run_per_model'] = 5
        train_hyperpara['train_valid_data_group'] = list(range(5)) + list(range(5))
        train_hyperpara['lr'] = 2e-5
        train_hyperpara['lr_decay'] = 1e-4
        train_hyperpara['patience'] = 1000
        train_hyperpara['task_order'] = list(range(data_hyperpara['num_tasks']))
        if args.task_order_type == 1:
            train_hyperpara['task_order'] = [0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        elif args.task_order_type == 2:
            train_hyperpara['task_order'] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]



        train_hyperpara['patience'] = 5
        train_hyperpara['num_run_per_model'] = 2





        train_hyperpara['learning_step_max'] = len(train_hyperpara['task_order']) * train_hyperpara['patience']

        train_data, validation_data, test_data = officehome_data(data_file_name, data_hyperpara['num_train_data'], data_hyperpara['num_valid_data'], data_hyperpara['num_test_data'], data_hyperpara['num_train_group'], data_hyperpara['image_dimension'])
        officehome_data_print_info(train_data, validation_data, test_data)
        classification_prob=True

    elif 'stl10' in data_type:
        data_hyperpara['multi_class_label'] = False
        data_hyperpara['image_dimension'] = [96, 96, 3]
        data_hyperpara['add_noise'] = True
        data_hyperpara['swap_channels'] = True

        if '15t' in data_type:
            data_hyperpara['num_tasks'] = 15
            data_hyperpara['validation_data_ratio'] = 50
            data_hyperpara['data_percent'] = 10
            data_hyperpara['num_classes'] = 5
        elif '20t' in data_type:
            data_hyperpara['num_tasks'] = 20
            data_hyperpara['validation_data_ratio'] = 15
            data_hyperpara['data_percent'] = 25
            data_hyperpara['num_classes'] = 3
        else:
            data_hyperpara['num_tasks'] = int(input("\tNumber of tasks: "))
            data_hyperpara['validation_data_ratio'] = int(input("\tRatio of validation data: "))
            data_hyperpara['data_percent'] = args.data_percent
            data_hyperpara['num_classes'] = args.num_classes

        data_file_name = 'stl10_p'+str(data_hyperpara['data_percent'])+'_v'+str(data_hyperpara['validation_data_ratio'])+'_t'+str(data_hyperpara['num_tasks'])+'_c'+str(data_hyperpara['num_classes'])

        train_hyperpara['num_run_per_model'] = 5
        train_hyperpara['train_valid_data_group'] = list(range(5)) + list(range(5))
        train_hyperpara['lr'] = 1e-4
        train_hyperpara['lr_decay'] = 0.0
        train_hyperpara['patience'] = 500
        train_hyperpara['task_order'] = list(range(data_hyperpara['num_tasks']))
        if args.task_order_type > 0:
            raise ValueError("Not specified the sequence of sub-tasks!")



        train_hyperpara['patience'] = 5
        train_hyperpara['num_run_per_model'] = 2




        train_hyperpara['learning_step_max'] = len(train_hyperpara['task_order']) * train_hyperpara['patience']

        categorized_train_data, categorized_test_data, experiments_design = stl10_data(experiment_file_base_name=data_file_name, valid_data_ratio_to_whole=float(data_hyperpara['validation_data_ratio'])/100.0, num_train_group=data_hyperpara['num_train_group'], num_tasks=data_hyperpara['num_tasks'], data_percent=float(data_hyperpara['data_percent'])/100.0, num_classes_per_task=data_hyperpara['num_classes'], allowNoise=data_hyperpara['add_noise'], allowChannelSwap=data_hyperpara['swap_channels'])
        print_experiment_design(experiments_design, data_type, print_info=True)
        classification_prob = True

    else:
        raise ValueError("The given dataset has no pre-defined experiment design. Check dataset again!")

    train_hyperpara['em_analysis_maxC'], train_hyperpara['em_analysis_maxC_prob'] = args.em_fix_maxC, args.em_maxP

    ## Model Set-up
    model_architecture, model_hyperpara = model_setup(data_type, data_hyperpara['image_dimension'], args.model_type, args.test_type, args.cnn_padtype_valid, args.skip_connect_test_type, args.highway_connect_test_type, args.num_clayers, args.phase1_max_epoch, args.darts_approx_order)
    train_hyperpara['num_tasks'] = data_hyperpara['num_tasks']

    saveparam = args.saveparam or 'bruteforce' in model_architecture

    save_param_path = None
    if saveparam:
        if not 'params' in listdir(getcwd()+'/Result'):
            mkdir('./Result/params')
        save_param_dir_name = data_type + '_' + str(data_percent) + 'p_' + args.model_type + '_t' + str(args.test_type)
        if args.highway_connect_test_type > 0:
            save_param_dir_name += '_h' + str(args.highway_connect_test_type)
        elif args.skip_connect_test_type > 0:
            save_param_dir_name += '_s' + str(args.skip_connect_test_type)
        while save_param_dir_name in listdir(getcwd()+'/Result/params'):
            save_param_dir_name += 'a'
        save_param_path = getcwd()+'/Result/params/'+save_param_dir_name
        mkdir(save_param_path)

    print(model_architecture)
    if ('tf' in model_architecture) and (args.tensorfactor_param_path is not None):
        tensorfactor_param_path = getcwd()+'/Result/'+args.tensorfactor_param_path
    else:
        tensorfactor_param_path = None
    if 'officehome' in data_type and args.task_order_type > 0:
        print("\tOrder of tasks ", train_hyperpara['task_order'])

    train_hyperpara['reset_prior'] = args.reset_prior

    ## Training the Model
    if 'stl10' in data_type:
        saved_result = train_run_for_each_model_v2(model_architecture, model_hyperpara, train_hyperpara, [categorized_train_data, categorized_test_data, experiments_design], data_type, mat_file_name, classification_prob, saved_result=None, useGPU=use_gpu, GPU_device=gpu_device_num, doLifelong=do_lifelong, saveParam=saveparam, saveParamDir=save_param_path, saveGraph=args.savegraph, tfInitParamPath=tensorfactor_param_path)
    else:
        saved_result = train_run_for_each_model(model_architecture, model_hyperpara, train_hyperpara, [train_data, validation_data, test_data], data_type, mat_file_name, classification_prob, saved_result=None, useGPU=use_gpu, GPU_device=gpu_device_num, doLifelong=do_lifelong, saveParam=saveparam, saveParamDir=save_param_path, saveGraph=args.savegraph, tfInitParamPath=tensorfactor_param_path)


if __name__ == '__main__':
    main()