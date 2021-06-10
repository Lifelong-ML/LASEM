import tensorflow as tf
import numpy as np
import shutil

from utils.utils import get_value_of_valid_tensors, savemat_wrapper, savemat_wrapper_nested_list, count_trainable_var2, get_list_of_valid_tensors, new_weight, new_bias
from utils.utils_nn import new_cnn_fc_net, new_hybrid_tensorfactored_cnn_fc_net, new_flexible_hardparam_cnn_fc_nets_ver2, new_TF_KB_param, new_TF_TS_param
from utils.utils_df_nn import new_ELLA_flexible_cnn_deconv_tensordot_fc_net, new_ELLA_KB_param
from classification.model.lifelong_model_frame import Lifelong_Model_BruteForceSearch_Frame

_tf_ver = tf.__version__.split('.')
_up_to_date_tf = int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) >= 14)
if _up_to_date_tf:
    _tf_tensor = tf.is_tensor
else:
    _tf_tensor = tf.contrib.framework.is_tensor

class LL_CNN_HPS_BruteForceSearch(Lifelong_Model_BruteForceSearch_Frame):
    def __init__(self, model_hyperpara, train_hyperpara, model_architecture):
        super().__init__(model_hyperpara, train_hyperpara, model_architecture)
        self.search_on_first_task = False

    def get_param(self, sess):
        def merge_sharing_config():
            result = [False for _ in range(self.num_conv_layers)]
            for elem in self.conv_sharing:
                for cnt in range(self.num_conv_layers):
                    result[cnt] = result[cnt] or elem[cnt]
            return result

        def convert_unnecessary_param_none(param, config, select_when_true_in_config=True):
            params_per_layer = len(param)//len(config)
            result = []
            for cnt in range(len(config)):
                for cnt2 in range(params_per_layer):
                    if (config[cnt] and select_when_true_in_config) or not (config[cnt] or select_when_true_in_config):
                        result.append(param[params_per_layer*cnt+cnt2])
                    else:
                        result.append(None)
            return result

        shared_cnn_param_val = convert_unnecessary_param_none(get_value_of_valid_tensors(sess, self.shared_conv_params), merge_sharing_config(), True)
        cnn_param_val = [convert_unnecessary_param_none(get_value_of_valid_tensors(sess, cnn_param), conf, False) for (cnn_param, conf) in zip(self.conv_params, self.conv_sharing)]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        parameters_val['conv_shared_weights'] = savemat_wrapper(shared_cnn_param_val)
        parameters_val['conv_taskspecific_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val

    def postprocess_loaded_params(self, loaded_params):
        def none_detector(np_arr):
            arr_shape = np_arr.shape
            if len(arr_shape) == 2 and arr_shape[0] == 1 and arr_shape[1] == 1:
                return None
            elif len(arr_shape) == 2 and (arr_shape[0] == 1 or arr_shape[1] == 1):
                return np.squeeze(np_arr)
            else:
                return np_arr

        conv_shared = loaded_params['conv_shared_weights'][0][0]
        conv_TS = loaded_params['conv_taskspecific_weights'][0][0]
        fc = loaded_params['fc_weights'][0][0]

        assert (conv_TS.shape[1]==fc.shape[1]), "Number of tasks within parameters are mis-matched!"
        num_tasks = conv_TS.shape[1]
        return_conv_shared, return_conv_TS, return_fc = [], [], []
        for cnt_clayer in range(conv_shared.shape[1]):
            return_conv_shared.append(none_detector(conv_shared[0, cnt_clayer]))

        for cnt_task in range(num_tasks):
            task_conv_TS, task_fc = [], []
            for cnt_clayer in range(conv_TS[0, cnt_task].shape[1]):
                task_conv_TS.append(none_detector(conv_TS[0, cnt_task][0, cnt_clayer]))

            for cnt_flayer in range(fc[0, cnt_task].shape[1]):
                task_fc.append(none_detector(fc[0, cnt_task][0, cnt_flayer]))
            return_conv_TS.append(task_conv_TS)
            return_fc.append(task_fc)

        params = {}
        params['conv_shared_weights'] = return_conv_shared
        params['conv_taskspecific_weights'] = return_conv_TS
        params['fc_weights'] = return_fc
        return params

    def _build_task_model(self, net_input, output_size, transfer_config, params=None, trainable=False):
        if params is None:
            params_conv, params_fc = None, None
        else:
            params_conv, params_fc = params['Conv'], params['FC']

        assert (len(self.shared_conv_params) == 2*self.num_conv_layers), "Given parameters of shared conv doesn't match the number of layers!"
        if params_conv is not None:
            assert (len(params_conv) == 2*self.num_conv_layers), "Given parameters of task-specific conv doesn't match the number of layers!"
        else:
            params_conv = [None for _ in range(2*self.num_conv_layers)]
        if params_fc is not None:
            assert (len(params_fc) == 2*self.num_fc_layers), "Given parameters of fc doesn't match the number of layers!"

        task_net, conv_params, _, fc_params = new_flexible_hardparam_cnn_fc_nets_ver2(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], transfer_config, cnn_activation_fn=self.hidden_act, shared_cnn_params=self.shared_conv_params, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, padding_type=self.padding_type, input_size=self.input_size[0:2], trainable=trainable, trainable_shared=True)
        return task_net, conv_params, fc_params

    def _build_whole_model(self, params=None):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            if (task_cnt==self.current_task and self.task_is_new) or params is None:
                params_to_use = None
            else:
                params_to_use = {'Conv': params['conv_taskspecific_weights'][task_cnt], 'FC': params['fc_weights'][task_cnt]}

            task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, self.conv_sharing[task_cnt], params=params_to_use, trainable=(task_cnt==self.current_task))
            self.task_models.append(task_net)
            self.conv_params.append(conv_params)
            self.fc_params.append(fc_params)
            self.params.append(conv_params+fc_params)
            self.num_trainable_var += count_trainable_var2(self.params[-1]) if task_cnt < 1 else count_trainable_var2(self.params[-1]) - self.shared_conv_params_size

        self.shared_conv_trainable_param = get_list_of_valid_tensors(self.shared_conv_params)
        self.conv_trainable_param = get_list_of_valid_tensors(self.conv_params[self.current_task])
        self.fc_trainable_param = get_list_of_valid_tensors(self.fc_params[self.current_task])
        self.trainable_params = list(self.shared_conv_params) + list(self.conv_trainable_param) + list(self.fc_trainable_param)

    def _shared_param_init(self, params_init=None):
        shared_conv_init_val = [None for _ in range(2*self.num_conv_layers)] if params_init is None else params_init['conv_shared_weights']
        self.shared_conv_params = []
        for layer_cnt in range(self.num_conv_layers):
            self.shared_conv_params.append(new_weight(shape=self.cnn_kernel_size[2*layer_cnt:2*(layer_cnt+1)]+self.cnn_channels_size[layer_cnt:layer_cnt+2], init_tensor=shared_conv_init_val[2*layer_cnt], trainable=True, name='Shared_Conv_W%d'%(layer_cnt)))
            self.shared_conv_params.append(new_bias(shape=[self.cnn_channels_size[layer_cnt+1]], init_tensor=shared_conv_init_val[2*layer_cnt+1], trainable=True, name='Shared_Conv_b%d'%(layer_cnt)))
        self.shared_conv_params_size = count_trainable_var2(self.shared_conv_params)

    def _copy_TS_to_shared(self, src_params, shared_config):
        new_shared_conv = []
        for layer_cnt, (conf) in enumerate(shared_config):
            if conf:
                new_shared_conv.append(src_params[2*layer_cnt])
                new_shared_conv.append(src_params[2*layer_cnt+1])
            else:
                new_shared_conv.append(None)
                new_shared_conv.append(None)
        return new_shared_conv

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False, params=None):
        if self.num_tasks == 1 and self.task_is_new:
            self.conv_sharing.append([False for _ in range(self.num_conv_layers)])
        elif self.num_tasks == 2 and self.task_is_new:
            self.conv_sharing[0] = self.conv_sharing[1]
            params['conv_shared_weights'] = self._copy_TS_to_shared(params['conv_taskspecific_weights'][0], self.conv_sharing[0])
        self._shared_param_init(params_init=params)
        super().add_new_task(output_dim, curr_task_index, single_input_placeholder, params)

    def postprocess_config_search(self, validation_configwise_accuracy, search_param_dir):
        best_conf = np.argmax(validation_configwise_accuracy)
        if self.num_tasks == 2 and self.task_is_new:
            self.conv_sharing[0] = self._possible_configs[best_conf]
            self.conv_sharing[1] = self._possible_configs[best_conf]
        else:
            self.conv_sharing[self.current_task] = self._possible_configs[best_conf]
        shutil.copy2(search_param_dir+'/parameter_config%d.mat'%(best_conf), self.save_param_dir+'/model_parameter_task%d.mat'%(self.task_indices[-1]))
        loaded_best_summary = self.load_training_summary(search_param_dir+'/summary_config%d.mat'%(best_conf))
        return loaded_best_summary['history_train_error'], loaded_best_summary['history_validation_error'], loaded_best_summary['history_test_error'], loaded_best_summary['history_best_test_error']


class LL_CNN_TF_BruteForceSearch(Lifelong_Model_BruteForceSearch_Frame):
    def __init__(self, model_hyperpara, train_hyperpara, model_architecture):
        super().__init__(model_hyperpara, train_hyperpara, model_architecture)
        self.aux_loss_weight = model_hyperpara['auxiliary_loss_weight']
        self.search_on_first_task = False

    def get_param(self, sess):
        def merge_sharing_config():
            result = [False for _ in range(self.num_conv_layers)]
            for elem in self.conv_sharing:
                for cnt in range(self.num_conv_layers):
                    result[cnt] = result[cnt] or elem[cnt]
            return result

        def convert_unnecessary_param_none(param, config, select_when_true_in_config=True):
            params_per_layer = len(param)//len(config)
            result = []
            for cnt in range(len(config)):
                for cnt2 in range(params_per_layer):
                    if (config[cnt] and select_when_true_in_config) or not (config[cnt] or select_when_true_in_config):
                        result.append(param[params_per_layer*cnt+cnt2])
                    else:
                        result.append(None)
            return result

        tf_KB_param_val = convert_unnecessary_param_none(get_value_of_valid_tensors(sess, self.tf_KB_params), merge_sharing_config(), True)
        tf_TS_param_val = [convert_unnecessary_param_none(get_value_of_valid_tensors(sess, cnn_TS_param), conf, True) for (cnn_TS_param, conf) in zip(self.tf_TS_params, self.conv_sharing)]
        cnn_gen_param_val = [convert_unnecessary_param_none(get_value_of_valid_tensors(sess, cnn_gen_param), conf, True) for (cnn_gen_param, conf) in zip(self.tf_gen_conv_params, self.conv_sharing)]
        cnn_param_val = [convert_unnecessary_param_none(get_value_of_valid_tensors(sess, cnn_param), conf, False) for (cnn_param, conf) in zip(self.conv_params, self.conv_sharing)]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        parameters_val['TF_KB'] = savemat_wrapper(tf_KB_param_val)
        parameters_val['TF_TS'] = savemat_wrapper_nested_list(tf_TS_param_val)
        parameters_val['conv_gen_weights'] = savemat_wrapper_nested_list(cnn_gen_param_val)
        parameters_val['conv_taskspecific_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val

    def postprocess_loaded_params(self, loaded_params):
        def none_detector(np_arr):
            arr_shape = np_arr.shape
            if len(arr_shape) == 2 and arr_shape[0] == 1 and arr_shape[1] == 1:
                return None
            elif len(arr_shape) == 2 and (arr_shape[0] == 1 or arr_shape[1] == 1):
                return np.squeeze(np_arr)
            else:
                return np_arr

        tf_KB = loaded_params['TF_KB'][0][0]
        tf_TS = loaded_params['TF_TS'][0][0]
        conv_gen = loaded_params['conv_gen_weights'][0][0]
        conv_TS = loaded_params['conv_taskspecific_weights'][0][0]
        fc = loaded_params['fc_weights'][0][0]

        assert(tf_TS.shape[1]==conv_gen.shape[1] and conv_gen.shape[1]==conv_TS.shape[1] and conv_TS.shape[1]==fc.shape[1]), "Number of tasks within parameters are mis-matched!"
        num_tasks = tf_TS.shape[1]
        return_tf_KB, return_tf_TS, return_conv_gen, return_conv_TS, return_fc = [], [], [], [], []
        for cnt_clayer in range(tf_KB.shape[1]):
            return_tf_KB.append(none_detector(tf_KB[0, cnt_clayer]))

        for cnt_task in range(num_tasks):
            task_tf_TS, task_conv_gen, task_conv_TS, task_fc = [], [], [], []

            for cnt_clayer in range(tf_TS[0, cnt_task].shape[1]):
                task_tf_TS.append(none_detector(tf_TS[0, cnt_task][0, cnt_clayer]))

            for cnt_clayer in range(conv_gen[0, cnt_task].shape[1]):
                task_conv_gen.append(none_detector(conv_gen[0, cnt_task][0, cnt_clayer]))

            for cnt_clayer in range(conv_TS[0, cnt_task].shape[1]):
                task_conv_TS.append(none_detector(conv_TS[0, cnt_task][0, cnt_clayer]))

            for cnt_flayer in range(fc[0, cnt_task].shape[1]):
                task_fc.append(none_detector(fc[0, cnt_task][0, cnt_flayer]))
            return_tf_TS.append(task_tf_TS)
            return_conv_gen.append(task_conv_gen)
            return_conv_TS.append(task_conv_TS)
            return_fc.append(task_fc)

        params = {}
        params['TF_KB'] = return_tf_KB
        params['TF_TS'] = return_tf_TS
        params['conv_gen_weights'] = return_conv_gen
        params['conv_taskspecific_weights'] = return_conv_TS
        params['fc_weights'] = return_fc
        return params

    def _build_task_model(self, net_input, output_size, transfer_config, task_cnt, params=None, trainable=False):
        if params is None:
            params_TS, params_conv, params_fc = None, None, None
        else:
            params_TS, params_conv, params_fc = params['TF_TS'], params['Conv'], params['FC']

        assert (len(self.tf_KB_params) == self.num_conv_layers), "Given parameters of DF KB doesn't match the number of layers!"
        if params_TS is not None:
            assert (len(params_TS) == 5*self.num_conv_layers), "Given trained parameters of DF TS doesn't match to the hyper-parameters!"
        if params_conv is not None:
            assert (len(params_conv) == 2*self.num_conv_layers), "Given trained parameters of conv doesn't match to the hyper-parameters!"
        if params_fc is not None:
            assert (len(params_fc) == 2*self.num_fc_layers), "Given trained parameters of fc doesn't match to the hyper-parameters!"

        task_net, _, tf_TS_param_tmp, gen_conv_param_tmp, conv_params, _, fc_params = new_hybrid_tensorfactored_cnn_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], transfer_config, cnn_activation_fn=self.hidden_act, cnn_KB_params=self.tf_KB_params, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable)
        self.tf_TS_params.append(tf_TS_param_tmp)
        self.tf_gen_conv_params.append(gen_conv_param_tmp)
        return task_net, conv_params, fc_params

    def _build_whole_model(self, params=None):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            if (task_cnt==self.current_task) and (self.task_is_new):
                param_to_use = None
            else:
                param_to_use = {'TF_TS': params['TF_TS'][task_cnt], 'Conv': params['conv_taskspecific_weights'][task_cnt], 'FC': params['fc_weights'][task_cnt]}
            task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, self.conv_sharing[task_cnt], task_cnt, params=param_to_use, trainable=(task_cnt==self.current_task))

            self.task_models.append(task_net)
            self.conv_params.append(conv_params)
            self.fc_params.append(fc_params)
            self.params.append(list(self.tf_KB_params)+list(self.tf_TS_params[-1])+list(conv_params)+list(fc_params))
            self.num_trainable_var += count_trainable_var2(self.params[-1]) if task_cnt < 1 else count_trainable_var2(self.params[-1]) - self.tf_KB_params_size

        self.tf_KB_trainable_param = get_list_of_valid_tensors(self.tf_KB_params)
        self.tf_TS_trainable_param = get_list_of_valid_tensors(self.tf_TS_params[self.current_task])
        self.conv_trainable_param = get_list_of_valid_tensors(self.conv_params[self.current_task])
        self.fc_trainable_param = get_list_of_valid_tensors(self.fc_params[self.current_task])
        self.trainable_params = list(self.tf_KB_trainable_param) + list(self.tf_TS_trainable_param) + list(self.conv_trainable_param) + list(self.fc_trainable_param)

    def _shared_param_init(self, params_init=None):
        self.tf_TS_params, self.tf_gen_conv_params = [], []
        KB_init_val = [None for _ in range(self.num_conv_layers)] if params_init is None else params_init['TF_KB']
        self.tf_KB_params = [new_TF_KB_param(self.cnn_kernel_size[2*layer_cnt:2*(layer_cnt+1)]+self.cnn_channels_size[layer_cnt:layer_cnt+2], layer_cnt, KB_init_val[layer_cnt], True) for layer_cnt in range(self.num_conv_layers)]
        self.tf_KB_params_size = count_trainable_var2(self.tf_KB_params)

    def _copy_shared_to_TS(self, src_params, shared_config):
        new_taskspecific_conv = []
        for layer_cnt, (conf) in enumerate(shared_config):
            if conf:
                new_taskspecific_conv.append(None)
                new_taskspecific_conv.append(None)
            else:
                new_taskspecific_conv.append(src_params[2*layer_cnt])
                new_taskspecific_conv.append(src_params[2*layer_cnt+1])
        return new_taskspecific_conv

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False, params=None):
        if self.num_tasks == 1 and self.task_is_new:
            self.conv_sharing.append([True for _ in range(self.num_conv_layers)])
        elif self.num_tasks == 2 and self.task_is_new:
            self.conv_sharing[0] = self.conv_sharing[1]
            params['conv_taskspecific_weights'][0] = self._copy_shared_to_TS(params['conv_gen_weights'][0], self.conv_sharing[0])
        self._shared_param_init(params_init=params)
        super().add_new_task(output_dim, curr_task_index, single_input_placeholder, params)

    def define_opt(self):
        def _exclude_bias_in_TS_param_list(param_list):
            # assume bias of conv layer is a vector (1-dim)
            params_to_return = []
            for p in param_list:
                p_dim = p.get_shape()
                if len(p_dim)>1:
                    params_to_return.append(p)
            return params_to_return

        def _compute_aux_loss(param_list, list_form=False):
            aux_loss_list = []
            for p in param_list:
                ts_dim = int(p.get_shape()[0])
                tensor_for_norm = tf.matmul(tf.transpose(p), p)-tf.eye(ts_dim)
                aux_loss_list.append(tf.norm(tensor_for_norm, ord='fro', axis=(0, 1)))

            if list_form:
                return aux_loss_list
            else:
                aux_loss = _sum_tensors_list(aux_loss_list)
                return aux_loss

        def _sum_tensors_list(tensor_list):
            sum = 0.0
            for t in tensor_list:
                sum += t
            return sum

        with tf.name_scope('Optimization'):
            aux_loss = _compute_aux_loss(_exclude_bias_in_TS_param_list(self.tf_TS_trainable_param))

            KB_grads = tf.gradients(self.loss[self.current_task], self.tf_KB_trainable_param)
            KB_grads_vars = [(grad, param) for grad, param in zip(KB_grads, self.tf_KB_trainable_param)]

            TS_grads = tf.gradients(self.loss[self.current_task]+self.aux_loss_weight*aux_loss, self.tf_TS_trainable_param)
            TS_grads_vars = [(grad, param) for grad, param in zip(TS_grads, self.tf_TS_trainable_param)]

            conv_grads = tf.gradients(self.loss[self.current_task], self.conv_trainable_param)
            conv_grads_vars = [(grad, param) for grad, param in zip(conv_grads, self.conv_trainable_param)]

            fc_grads = tf.gradients(self.loss[self.current_task], self.fc_trainable_param)
            fc_grads_vars = [(grad, param) for grad, param in zip(fc_grads, self.fc_trainable_param)]

            self.grads = list(KB_grads)+list(TS_grads)+list(conv_grads)+list(fc_grads)
            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            grads_vars = KB_grads_vars+TS_grads_vars+conv_grads_vars+fc_grads_vars
            self.update = trainer.apply_gradients(grads_vars)

    def postprocess_config_search(self, validation_configwise_accuracy, search_param_dir):
        best_conf = np.argmax(validation_configwise_accuracy)
        if self.num_tasks == 2 and self.task_is_new:
            self.conv_sharing[0] = self._possible_configs[best_conf]
            self.conv_sharing[1] = self._possible_configs[best_conf]
        else:
            self.conv_sharing[self.current_task] = self._possible_configs[best_conf]
        shutil.copy2(search_param_dir+'/parameter_config%d.mat'%(best_conf), self.save_param_dir+'/model_parameter_task%d.mat'%(self.task_indices[-1]))
        loaded_best_summary = self.load_training_summary(search_param_dir+'/summary_config%d.mat'%(best_conf))
        return loaded_best_summary['history_train_error'], loaded_best_summary['history_validation_error'], loaded_best_summary['history_test_error'], loaded_best_summary['history_best_test_error']


class LL_CNN_DFCNN_BruteForceSearch(Lifelong_Model_BruteForceSearch_Frame):
    def __init__(self, model_hyperpara, train_hyperpara, model_architecture):
        super().__init__(model_hyperpara, train_hyperpara, model_architecture)
        self.dfcnn_KB_size = model_hyperpara['cnn_KB_sizes']
        self.dfcnn_TS_size = model_hyperpara['cnn_TS_sizes']
        self.dfcnn_stride_size = model_hyperpara['cnn_deconv_stride_sizes']
        self.dfcnn_KB_reg_scale = model_hyperpara['regularization_scale'][1]
        self.dfcnn_TS_reg_scale = model_hyperpara['regularization_scale'][3]
        self.search_on_first_task = False

    def get_param(self, sess):
        def merge_sharing_config():
            result = [False for _ in range(self.num_conv_layers)]
            for elem in self.conv_sharing:
                for cnt in range(self.num_conv_layers):
                    result[cnt] = result[cnt] or elem[cnt]
            return result

        def convert_unnecessary_param_none(param, config, select_when_true_in_config=True):
            params_per_layer = len(param)//len(config)
            result = []
            for cnt in range(len(config)):
                for cnt2 in range(params_per_layer):
                    if (config[cnt] and select_when_true_in_config) or not (config[cnt] or select_when_true_in_config):
                        result.append(param[params_per_layer*cnt+cnt2])
                    else:
                        result.append(None)
            return result

        dfcnn_KB_param_val = convert_unnecessary_param_none(get_value_of_valid_tensors(sess, self.dfcnn_KB_params), merge_sharing_config(), True)
        dfcnn_TS_param_val = [convert_unnecessary_param_none(get_value_of_valid_tensors(sess, cnn_TS_param), conf, True) for (cnn_TS_param, conf) in zip(self.dfcnn_TS_params, self.conv_sharing)]
        cnn_gen_param_val = [convert_unnecessary_param_none(get_value_of_valid_tensors(sess, cnn_gen_param), conf, True) for (cnn_gen_param, conf) in zip(self.dfcnn_gen_conv_params, self.conv_sharing)]
        cnn_param_val = [convert_unnecessary_param_none(get_value_of_valid_tensors(sess, cnn_param), conf, False) for (cnn_param, conf) in zip(self.conv_params, self.conv_sharing)]
        fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        parameters_val['DFCNN_KB'] = savemat_wrapper(dfcnn_KB_param_val)
        parameters_val['DFCNN_TS'] = savemat_wrapper_nested_list(dfcnn_TS_param_val)
        parameters_val['conv_gen_weights'] = savemat_wrapper_nested_list(cnn_gen_param_val)
        parameters_val['conv_taskspecific_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val

    def postprocess_loaded_params(self, loaded_params):
        def none_detector(np_arr):
            arr_shape = np_arr.shape
            if len(arr_shape) == 2 and arr_shape[0] == 1 and arr_shape[1] == 1:
                return None
            elif len(arr_shape) == 2 and (arr_shape[0] == 1 or arr_shape[1] == 1):
                return np.squeeze(np_arr)
            else:
                return np_arr

        dfcnn_KB = loaded_params['DFCNN_KB'][0][0]
        dfcnn_TS = loaded_params['DFCNN_TS'][0][0]
        conv_gen = loaded_params['conv_gen_weights'][0][0]
        conv_TS = loaded_params['conv_taskspecific_weights'][0][0]
        fc = loaded_params['fc_weights'][0][0]

        assert(dfcnn_TS.shape[1]==conv_gen.shape[1] and conv_gen.shape[1]==conv_TS.shape[1] and conv_TS.shape[1]==fc.shape[1]), "Number of tasks within parameters are mis-matched!"
        num_tasks = dfcnn_TS.shape[1]
        return_dfcnn_KB, return_dfcnn_TS, return_conv_gen, return_conv_TS, return_fc = [], [], [], [], []
        for cnt_clayer in range(dfcnn_KB.shape[1]):
            return_dfcnn_KB.append(none_detector(dfcnn_KB[0, cnt_clayer]))

        for cnt_task in range(num_tasks):
            task_dfcnn_TS, task_conv_gen, task_conv_TS, task_fc = [], [], [], []

            for cnt_clayer in range(dfcnn_TS[0, cnt_task].shape[1]):
                task_dfcnn_TS.append(none_detector(dfcnn_TS[0, cnt_task][0, cnt_clayer]))

            for cnt_clayer in range(conv_gen[0, cnt_task].shape[1]):
                task_conv_gen.append(none_detector(conv_gen[0, cnt_task][0, cnt_clayer]))

            for cnt_clayer in range(conv_TS[0, cnt_task].shape[1]):
                task_conv_TS.append(none_detector(conv_TS[0, cnt_task][0, cnt_clayer]))

            for cnt_flayer in range(fc[0, cnt_task].shape[1]):
                task_fc.append(none_detector(fc[0, cnt_task][0, cnt_flayer]))
            return_dfcnn_TS.append(task_dfcnn_TS)
            return_conv_gen.append(task_conv_gen)
            return_conv_TS.append(task_conv_TS)
            return_fc.append(task_fc)

        params = {}
        params['DFCNN_KB'] = return_dfcnn_KB
        params['DFCNN_TS'] = return_dfcnn_TS
        params['conv_gen_weights'] = return_conv_gen
        params['conv_taskspecific_weights'] = return_conv_TS
        params['fc_weights'] = return_fc
        return params

    def _build_task_model(self, net_input, output_size, transfer_config, task_cnt, params=None, trainable=False):
        if params is None:
            params_TS, params_conv, params_fc = None, None, None
        else:
            params_TS, params_conv, params_fc = params['DFCNN_TS'], params['Conv'], params['FC']

        assert (len(self.dfcnn_KB_params) == self.num_conv_layers), "Given parameters of DF KB doesn't match the number of layers!"
        if params_TS is not None:
            assert (len(params_TS) == 4*self.num_conv_layers), "Given trained parameters of DF TS doesn't match to the hyper-parameters!"
        if params_conv is not None:
            assert (len(params_conv) == 2*self.num_conv_layers), "Given trained parameters of conv doesn't match to the hyper-parameters!"
        if params_fc is not None:
            assert (len(params_fc) == 2*self.num_fc_layers), "Given trained parameters of fc doesn't match to the hyper-parameters!"

        task_net, _, dfcnn_TS_param_tmp, gen_conv_param_tmp, conv_params, _, fc_params = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], transfer_config, self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=self.dfcnn_KB_params, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable)
        self.dfcnn_TS_params.append(dfcnn_TS_param_tmp)
        self.dfcnn_gen_conv_params.append(gen_conv_param_tmp)
        return task_net, conv_params, fc_params

    def _build_whole_model(self, params=None):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            if (task_cnt==self.current_task) and (self.task_is_new):
                param_to_use = None
            else:
                param_to_use = {'DFCNN_TS': params['DFCNN_TS'][task_cnt], 'Conv': params['conv_taskspecific_weights'][task_cnt], 'FC': params['fc_weights'][task_cnt]}
            task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, self.conv_sharing[task_cnt], task_cnt, params=param_to_use, trainable=(task_cnt==self.current_task))

            self.task_models.append(task_net)
            self.conv_params.append(conv_params)
            self.fc_params.append(fc_params)
            self.params.append(list(self.dfcnn_KB_params)+list(self.dfcnn_TS_params[-1])+list(conv_params)+list(fc_params))
            self.num_trainable_var += count_trainable_var2(self.params[-1]) if task_cnt < 1 else count_trainable_var2(self.params[-1]) - self.dfcnn_KB_params_size

        self.dfcnn_KB_trainable_param = get_list_of_valid_tensors(self.dfcnn_KB_params)
        self.dfcnn_TS_trainable_param = get_list_of_valid_tensors(self.dfcnn_TS_params[self.current_task])
        self.conv_trainable_param = get_list_of_valid_tensors(self.conv_params[self.current_task])
        self.fc_trainable_param = get_list_of_valid_tensors(self.fc_params[self.current_task])
        self.trainable_params = list(self.dfcnn_KB_trainable_param) + list(self.dfcnn_TS_trainable_param) + list(self.conv_trainable_param) + list(self.fc_trainable_param)

    def _shared_param_init(self, params_init=None):
        self.dfcnn_TS_params, self.dfcnn_gen_conv_params = [], []
        self.KB_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_KB_reg_scale)
        self.TS_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_TS_reg_scale)
        KB_init_val = [None for _ in range(self.num_conv_layers)] if params_init is None else params_init['DFCNN_KB']
        self.dfcnn_KB_params = [new_ELLA_KB_param([1, self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt+1]], layer_cnt, 0, self.KB_l2_reg, KB_init_val[layer_cnt], True) for layer_cnt in range(self.num_conv_layers)]
        self.dfcnn_KB_params_size = count_trainable_var2(self.dfcnn_KB_params)

    def _copy_shared_to_TS(self, src_params, shared_config):
        new_taskspecific_conv = []
        for layer_cnt, (conf) in enumerate(shared_config):
            if conf:
                new_taskspecific_conv.append(None)
                new_taskspecific_conv.append(None)
            else:
                new_taskspecific_conv.append(src_params[2*layer_cnt])
                new_taskspecific_conv.append(src_params[2*layer_cnt+1])
        return new_taskspecific_conv

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False, params=None):
        if self.num_tasks == 1 and self.task_is_new:
            self.conv_sharing.append([True for _ in range(self.num_conv_layers)])
        elif self.num_tasks == 2 and self.task_is_new:
            self.conv_sharing[0] = self.conv_sharing[1]
            params['conv_taskspecific_weights'][0] = self._copy_shared_to_TS(params['conv_gen_weights'][0], self.conv_sharing[0])
        self._shared_param_init(params_init=params)
        super().add_new_task(output_dim, curr_task_index, single_input_placeholder, params)

    def define_opt(self):
        with tf.name_scope('Optimization'):
            reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            KB_reg_term2 = tf.contrib.layers.apply_regularization(self.KB_l2_reg, reg_var)
            TS_reg_term2 = tf.contrib.layers.apply_regularization(self.TS_l2_reg, reg_var)

            KB_grads = tf.gradients(self.loss[self.current_task]+KB_reg_term2, self.dfcnn_KB_trainable_param)
            KB_grads_vars = [(grad, param) for grad, param in zip(KB_grads, self.dfcnn_KB_trainable_param)]

            TS_grads = tf.gradients(self.loss[self.current_task]+TS_reg_term2, self.dfcnn_TS_trainable_param)
            TS_grads_vars = [(grad, param) for grad, param in zip(TS_grads, self.dfcnn_TS_trainable_param)]

            conv_grads = tf.gradients(self.loss[self.current_task], self.conv_trainable_param)
            conv_grads_vars = [(grad, param) for grad, param in zip(conv_grads, self.conv_trainable_param)]

            fc_grads = tf.gradients(self.loss[self.current_task], self.fc_trainable_param)
            fc_grads_vars = [(grad, param) for grad, param in zip(fc_grads, self.fc_trainable_param)]

            self.grads = list(KB_grads)+list(TS_grads)+list(conv_grads)+list(fc_grads)
            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            grads_vars = KB_grads_vars+TS_grads_vars+conv_grads_vars+fc_grads_vars
            self.update = trainer.apply_gradients(grads_vars)

    def postprocess_config_search(self, validation_configwise_accuracy, search_param_dir):
        best_conf = np.argmax(validation_configwise_accuracy)
        if self.num_tasks == 2 and self.task_is_new:
            self.conv_sharing[0] = self._possible_configs[best_conf]
            self.conv_sharing[1] = self._possible_configs[best_conf]
        else:
            self.conv_sharing[self.current_task] = self._possible_configs[best_conf]
        shutil.copy2(search_param_dir+'/parameter_config%d.mat'%(best_conf), self.save_param_dir+'/model_parameter_task%d.mat'%(self.task_indices[-1]))
        loaded_best_summary = self.load_training_summary(search_param_dir+'/summary_config%d.mat'%(best_conf))
        return loaded_best_summary['history_train_error'], loaded_best_summary['history_validation_error'], loaded_best_summary['history_test_error'], loaded_best_summary['history_best_test_error']