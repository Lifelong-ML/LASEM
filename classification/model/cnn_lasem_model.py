import tensorflow as tf
import numpy as np

from utils.utils import get_list_of_valid_tensors, get_value_of_valid_tensors, savemat_wrapper, savemat_wrapper_nested_list, count_trainable_var2
from utils.utils import new_weight, new_bias
from utils.utils_nn import new_flexible_hardparam_cnn_fc_nets_ver2, new_hybrid_tensorfactored_cnn_fc_net, new_TF_KB_param, new_TF_TS_param
from utils.utils_df_nn import new_ELLA_flexible_cnn_deconv_tensordot_fc_net, new_ELLA_KB_param, new_ELLA_cnn_deconv_tensordot_TS_param
from classification.model.lifelong_model_frame import Lifelong_Model_EM_Algo_Frame

_tf_ver = tf.__version__.split('.')
_up_to_date_tf = int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) >= 14)
if _up_to_date_tf:
    _tf_tensor = tf.is_tensor
else:
    _tf_tensor = tf.contrib.framework.is_tensor


########################################################################################################################
#####                    Apply EM algorithm to learn layers to share in baseline lifelong model                    #####
########################################################################################################################

#### Hard-Parameter Sharing + EM
class LL_CNN_HPS_EM_algo(Lifelong_Model_EM_Algo_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)

    def _build_task_model(self, net_input, output_size, task_cnt, params=None, trainable=False):
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

        if (task_cnt==self.current_task) and self.task_is_new:
            ## New task model => use EM algorithm
            with tf.name_scope('Hybrid_HPS_auto_EM'):
                ### 1. generate task-specific parameters in advance
                conv_params = []
                for layer_cnt in range(self.num_conv_layers):
                    conv_params.append(new_weight(shape=self.cnn_kernel_size[2*layer_cnt:2*(layer_cnt+1)]+self.cnn_channels_size[layer_cnt:layer_cnt+2], init_tensor=params_conv[2*layer_cnt], trainable=trainable, name='Conv_W%d'%(layer_cnt)))
                    conv_params.append(new_bias(shape=[self.cnn_channels_size[layer_cnt+1]], init_tensor=params_conv[2*layer_cnt+1], trainable=trainable, name='Conv_b%d'%(layer_cnt)))

                ### 2. generate task_models based on configs in self._possible_configs
                task_net, fc_params = [], None
                for conf in self._possible_configs:
                    net_tmp, _, _, fc_params = new_flexible_hardparam_cnn_fc_nets_ver2(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], conf, cnn_activation_fn=self.hidden_act, shared_cnn_params=self.shared_conv_params, cnn_params=conv_params, fc_activation_fn=self.hidden_act, fc_params=fc_params, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, padding_type=self.padding_type, input_size=self.input_size[0:2], trainable=trainable, trainable_shared=True, skip_connections=list(self.skip_connect))
                    task_net.append(net_tmp[-1])
        else:
            ## Hybrid HPS with the learned configuration
            task_net, conv_params, _, fc_params = new_flexible_hardparam_cnn_fc_nets_ver2(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], self.conv_sharing[task_cnt], cnn_activation_fn=self.hidden_act, shared_cnn_params=self.shared_conv_params, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, padding_type=self.padding_type, input_size=self.input_size[0:2], trainable=trainable, trainable_shared=True, skip_connections=list(self.skip_connect))
        return task_net, conv_params, fc_params

    def _build_whole_model(self):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, task_cnt, params=None if (task_cnt==self.current_task) and (self.task_is_new) else self.np_params[task_cnt], trainable=(task_cnt==self.current_task))
            self.task_models.append(task_net)
            self.conv_params.append(conv_params)
            self.fc_params.append(fc_params)
            self.params.append(conv_params+fc_params)
            self.num_trainable_var += count_trainable_var2(self.params[-1]) if task_cnt < 1 else count_trainable_var2(self.params[-1]) - self.shared_conv_params_size

        self.shared_conv_trainable_param = get_list_of_valid_tensors(self.shared_conv_params)
        self.conv_trainable_param = get_list_of_valid_tensors(self.conv_params[self.current_task])
        self.fc_trainable_param = get_list_of_valid_tensors(self.fc_params[self.current_task])
        self.trainable_params = list(self.shared_conv_params) + list(self.conv_trainable_param) + list(self.fc_trainable_param)

    def _shared_param_init(self):
        shared_conv_init_val = self.np_params[0]['Shared_Conv'] if hasattr(self, 'np_params') else [None for _ in range(2*self.num_conv_layers)]
        self.shared_conv_params = []
        for layer_cnt in range(self.num_conv_layers):
            self.shared_conv_params.append(new_weight(shape=self.cnn_kernel_size[2*layer_cnt:2*(layer_cnt+1)]+self.cnn_channels_size[layer_cnt:layer_cnt+2], init_tensor=shared_conv_init_val[2*layer_cnt], trainable=True, name='Shared_Conv_W%d'%(layer_cnt)))
            self.shared_conv_params.append(new_bias(shape=[self.cnn_channels_size[layer_cnt+1]], init_tensor=shared_conv_init_val[2*layer_cnt+1], trainable=True, name='Shared_Conv_b%d'%(layer_cnt)))
        self.shared_conv_params_size = count_trainable_var2(self.shared_conv_params)

    def define_opt(self):
        with tf.name_scope('Optimization'):
            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            if self.task_is_new:
                self.posterior_placeholder = tf.placeholder(tf.float32, shape=self.posterior.shape)
                self.update_prior = tf.assign(self.prior, self.posterior_placeholder)

                self.grads = [[] for _ in range(len(self.trainable_params))]
                for Loss in self.loss_for_train:
                    shared_conv_grads = tf.gradients(Loss, self.shared_conv_trainable_param)
                    conv_grads = tf.gradients(Loss, self.conv_trainable_param)
                    fc_grads = tf.gradients(Loss, self.fc_trainable_param)

                    collected_grads = list(shared_conv_grads) + list(conv_grads) + list(fc_grads)
                    for param_index, (c_g) in enumerate(collected_grads):
                        self.grads[param_index].append(c_g)

                weighted_summed_grads = []
                for grad_list in self.grads:
                    weighted_summed_grads.append(self._weighted_sum_grads(grad_list, self.posterior_placeholder))
                grads_vars = [(grad, param) for grad, param in zip(weighted_summed_grads, self.trainable_params)]
                self.update = trainer.apply_gradients(grads_vars)
            else:
                shared_conv_grads = tf.gradients(self.loss[self.current_task], self.shared_conv_trainable_param)
                shared_conv_grads_vars = [(grad, param) for grad, param in zip(shared_conv_grads, self.shared_conv_trainable_param)]

                conv_grads = tf.gradients(self.loss[self.current_task], self.conv_trainable_param)
                conv_grads_vars = [(grad, param) for grad, param in zip(conv_grads, self.conv_trainable_param)]

                fc_grads = tf.gradients(self.loss[self.current_task], self.fc_trainable_param)
                fc_grads_vars = [(grad, param) for grad, param in zip(fc_grads, self.fc_trainable_param)]

                self.grads = list(shared_conv_grads) + list(conv_grads) + list(fc_grads)
                grads_vars = shared_conv_grads_vars + conv_grads_vars + fc_grads_vars
                self.update = trainer.apply_gradients(grads_vars)

    def get_params_val(self, sess, use_npparams=True):
        if use_npparams:
            shared_cnn_param_val = self.np_params[0]['Shared_Conv']
            cnn_param_val = [np_p['Conv'] for np_p in self.np_params]
            fc_param_val = [np_p['FC'] for np_p in self.np_params]
        else:
            shared_cnn_param_val = get_value_of_valid_tensors(sess, self.shared_conv_params)
            cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.conv_params]
            fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        parameters_val['conv_shared_weights'] = savemat_wrapper(shared_cnn_param_val)
        parameters_val['conv_taskspecific_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val

    def convert_tfVar_to_npVar(self, sess):
        if (self.num_tasks == 1 and self.task_is_new):
            orig_shared_conv = [None for _ in range(2*self.num_conv_layers)]
        else:
            orig_shared_conv = list(self.np_params[0]['Shared_Conv'])

        def list_param_converter(list_of_params):
            converted_params = []
            for p in list_of_params:
                if type(p) == np.ndarray:
                    converted_params.append(p)
                elif _tf_tensor(p):
                    converted_params.append(sess.run(p))
                else:
                    converted_params.append(p)  ## append 'None' param
            return converted_params

        def double_list_param_converter(list_of_params):
            converted_params = []
            for task_params in list_of_params:
                converted_params.append(list_param_converter(task_params))
            return converted_params

        def post_process(layers_to_share, original_shared_conv, updated_shared_conv, updated_TS_conv):
            for layer_cnt, (sharing_flag) in enumerate(layers_to_share):
                if sharing_flag:
                    updated_TS_conv[self.current_task][2*layer_cnt], updated_TS_conv[self.current_task][2*layer_cnt+1] = None, None
                else:
                    updated_shared_conv[2*layer_cnt], updated_shared_conv[2*layer_cnt+1] = original_shared_conv[2*layer_cnt], original_shared_conv[2*layer_cnt+1]
            return updated_shared_conv, updated_TS_conv

        self.np_params = []
        np_shared_conv = list_param_converter(self.shared_conv_params)
        np_conv = double_list_param_converter(self.conv_params)
        np_fc = double_list_param_converter(self.fc_params)
        if self.task_is_new:
            learned_config = self._possible_configs[self.best_config(sess)]
            self.conv_sharing.append(learned_config)
            np_shared_conv, np_conv = post_process(learned_config, orig_shared_conv, np_shared_conv, np_conv)

        for cnt, (c, f) in enumerate(zip(np_conv, np_fc)):
            self.np_params.append({'Shared_Conv': np_shared_conv, 'Conv': c, 'FC': f} if cnt<1 else {'Conv': c, 'FC': f})

    def _collect_trainable_variables(self):
        return_list = []
        for p in self.shared_conv_params:
            if p is not None:
                return_list.append(p)
        for p in self.conv_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.fc_params[-1]:
            if p is not None:
                return_list.append(p)
        return return_list


######## Hybrid Tensor Factorized model + EM learning (Lifelong Learning ver.)
########          - based on Adrian Bulat, et al. Incremental Multi-domain Learning with Network Latent Tensor Factorization
class LL_hybrid_TF_EM_algo(Lifelong_Model_EM_Algo_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)
        self.aux_loss_weight = model_hyperpara['auxiliary_loss_weight']

    def _build_task_model(self, net_input, output_size, task_cnt, params=None, trainable=False):
        if params is None:
            params_KB, params_TS, params_conv, params_fc = None, None, None, None
        else:
            params_KB, params_TS, params_conv, params_fc = params['KB'], params['TS'], params['Conv'], params['FC']

        if params_KB is not None:
            assert (len(params_KB) == self.num_conv_layers), "Given trained parameters of DF KB doesn't match to the hyper-parameters!"
        if params_TS is not None:
            assert (len(params_TS) == 5*self.num_conv_layers), "Given trained parameters of DF TS doesn't match to the hyper-parameters!"
        if params_conv is not None:
            assert (len(params_conv) == 2*self.num_conv_layers), "Given trained parameters of conv doesn't match to the hyper-parameters!"
        else:
            params_conv = [None for _ in range(2*self.num_conv_layers)]
        if params_fc is not None:
            assert (len(params_fc) == 2*self.num_fc_layers), "Given trained parameters of fc doesn't match to the hyper-parameters!"
        else:
            params_fc = [None for _ in range(2*self.num_fc_layers)]

        if (task_cnt==self.current_task) and self.task_is_new:
            ## New task model => use EM algorithm
            ### 1. make parameters used in any of configs
            with tf.name_scope('Hybrid_TF_auto_EM'):
                conv_TS_param_tmp, conv_params = [], []
                for layer_cnt in range(self.num_conv_layers):
                    conv_TS_param_tmp += new_TF_TS_param([[self.cnn_kernel_size[2*layer_cnt], self.cnn_kernel_size[2*layer_cnt]], [self.cnn_kernel_size[2*layer_cnt+1], self.cnn_kernel_size[2*layer_cnt+1]], [self.cnn_channels_size[layer_cnt], self.cnn_channels_size[layer_cnt]], [self.cnn_channels_size[layer_cnt+1], self.cnn_channels_size[layer_cnt+1]], [self.cnn_channels_size[layer_cnt+1]]], layer_cnt, task_cnt, [None, None, None, None, None], trainable)
                    conv_params += [new_weight(shape=self.cnn_kernel_size[2*layer_cnt:2*(layer_cnt+1)]+self.cnn_channels_size[layer_cnt:layer_cnt+2], trainable=trainable, name='Conv_W%d'%(layer_cnt)), new_bias(shape=[self.cnn_channels_size[layer_cnt+1]], trainable=trainable, name='Conv_b%d'%(layer_cnt))]

                ### 2. generate task_models based on configs in self._possible_configs
                task_net, fc_params = [], None
                for conf in self._possible_configs:
                    net_tmp, _, _, _, _, _, fc_params = new_hybrid_tensorfactored_cnn_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], conf, cnn_activation_fn=self.hidden_act, cnn_KB_params=params_KB, cnn_TS_params=conv_TS_param_tmp, cnn_params=conv_params, fc_activation_fn=self.hidden_act, fc_params=fc_params, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable)
                    task_net.append(net_tmp[-1])
        else:
            ## Hybrid DF-CNN with the learned configuration
            task_net, _, conv_TS_param_tmp, gen_conv_param_tmp, conv_params, _, fc_params = new_hybrid_tensorfactored_cnn_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], self.conv_sharing[task_cnt], cnn_activation_fn=self.hidden_act, cnn_KB_params=params_KB, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable)
        self.conv_TS_params.append(conv_TS_param_tmp)
        #self.gen_conv_params.append(gen_conv_param_tmp)
        return task_net, conv_params, fc_params

    def _build_whole_model(self):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            if (task_cnt==self.current_task) and (self.task_is_new):
                param_to_reuse = {'KB': self.shared_conv_KB_params, 'TS': None, 'Conv': None, 'FC': None}
            else:
                param_to_reuse = {'KB': self.shared_conv_KB_params, 'TS': self.np_params[task_cnt]['TS'], 'Conv': self.np_params[task_cnt]['Conv'], 'FC': self.np_params[task_cnt]['FC']}
            task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, task_cnt, params=param_to_reuse, trainable=(task_cnt==self.current_task))

            if task_cnt == 0:
                self.shared_conv_KB_params_size = count_trainable_var2(self.shared_conv_KB_params)

            self.task_models.append(task_net)
            self.conv_params.append(conv_params)
            self.fc_params.append(fc_params)
            self.params.append(self._collect_trainable_variables())
            self.num_trainable_var += count_trainable_var2(self.params[-1]) if task_cnt < 1 else count_trainable_var2(self.params[-1]) - self.shared_conv_KB_params_size

        self.shared_conv_KB_trainable_param = get_list_of_valid_tensors(self.shared_conv_KB_params)
        self.conv_TS_trainable_param = get_list_of_valid_tensors(self.conv_TS_params[self.current_task])
        self.conv_trainable_param = get_list_of_valid_tensors(self.conv_params[self.current_task])
        self.fc_trainable_param = get_list_of_valid_tensors(self.fc_params[self.current_task])
        self.trainable_params = list(self.shared_conv_KB_trainable_param) + list(self.conv_TS_trainable_param) + list(self.conv_trainable_param) + list(self.fc_trainable_param)

    def _shared_param_init(self):
        self.conv_TS_params, self.gen_conv_params = [], []
        KB_init_val = self.np_params[0]['KB'] if hasattr(self, 'np_params') else [None for _ in range(self.num_conv_layers)]
        self.shared_conv_KB_params = [new_TF_KB_param(self.cnn_kernel_size[2*layer_cnt:2*(layer_cnt+1)]+self.cnn_channels_size[layer_cnt:layer_cnt+2], layer_cnt, KB_init_val[layer_cnt], True) for layer_cnt in range(self.num_conv_layers)]

    def _exclude_bias_in_TS_param_list(self, param_list):
        # assume bias of conv layer is a vector (1-dim)
        params_to_return = []
        for p in param_list:
            p_dim = p.get_shape()
            if len(p_dim) > 1:
                params_to_return.append(p)
        return params_to_return

    def _compute_aux_loss(self, param_list, list_form=False):
        aux_loss_list = []
        for p in param_list:
            ts_dim = int(p.get_shape()[0])
            tensor_for_norm = tf.matmul(tf.transpose(p), p) - tf.eye(ts_dim)
            aux_loss_list.append(tf.norm(tensor_for_norm, ord='fro', axis=(0, 1)))

        if list_form:
            return aux_loss_list
        else:
            aux_loss = self._sum_tensors_list(aux_loss_list)
            return aux_loss

    def _sum_tensors_list(self, tensor_list):
        sum = 0.0
        for t in tensor_list:
            sum += t
        return sum

    def define_opt(self):
        with tf.name_scope('Optimization'):
            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            if self.task_is_new:
                self.posterior_placeholder = tf.placeholder(tf.float32, shape=self.posterior.shape)
                self.update_prior = tf.assign(self.prior, self.posterior_placeholder)

                self.grads = [[] for _ in range(len(self.trainable_params))]
                aux_loss_per_param = self._compute_aux_loss(self._exclude_bias_in_TS_param_list(self.conv_TS_params[self.current_task]), list_form=True)
                for (Loss, conf) in zip(self.loss_for_train, self._possible_configs):
                    _, selected_aux_losses, _ = self._choose_params_for_sharing_config(None, aux_loss_per_param, None, conf, 4)
                    aux_loss = self._sum_tensors_list(selected_aux_losses)

                    KB_grads = tf.gradients(Loss, self.shared_conv_KB_trainable_param)
                    TS_grads = tf.gradients(Loss + self.aux_loss_weight*aux_loss, self.conv_TS_trainable_param)
                    conv_grads = tf.gradients(Loss, self.conv_trainable_param)
                    fc_grads = tf.gradients(Loss, self.fc_trainable_param)

                    collected_grads = list(KB_grads) + list(TS_grads) + list(conv_grads) + list(fc_grads)
                    for param_index, (c_g) in enumerate(collected_grads):
                        self.grads[param_index].append(c_g)

                weighted_summed_grads = []
                for grad_list in self.grads:
                    weighted_summed_grads.append(self._weighted_sum_grads(grad_list, self.posterior_placeholder))
                grads_vars = [(grad, param) for grad, param in zip(weighted_summed_grads, self.trainable_params)]
                self.update = trainer.apply_gradients(grads_vars)
            else:
                aux_loss = self._compute_aux_loss(self._exclude_bias_in_TS_param_list(self.conv_TS_trainable_param))

                KB_grads = tf.gradients(self.loss[self.current_task], self.shared_conv_KB_trainable_param)
                KB_grads_vars = [(grad, param) for grad, param in zip(KB_grads, self.shared_conv_KB_trainable_param)]

                TS_grads = tf.gradients(self.loss[self.current_task] + self.aux_loss_weight*aux_loss, self.conv_TS_trainable_param)
                TS_grads_vars = [(grad, param) for grad, param in zip(TS_grads, self.conv_TS_trainable_param)]

                conv_grads = tf.gradients(self.loss[self.current_task], self.conv_trainable_param)
                conv_grads_vars = [(grad, param) for grad, param in zip(conv_grads, self.conv_trainable_param)]

                fc_grads = tf.gradients(self.loss[self.current_task], self.fc_trainable_param)
                fc_grads_vars = [(grad, param) for grad, param in zip(fc_grads, self.fc_trainable_param)]

                self.grads = list(KB_grads) + list(TS_grads) + list(conv_grads) + list(fc_grads)
                grads_vars = KB_grads_vars + TS_grads_vars + conv_grads_vars + fc_grads_vars
                self.update = trainer.apply_gradients(grads_vars)

    def get_params_val(self, sess, use_npparams=True):
        if use_npparams:
            KB_param_val = self.np_params[0]['KB']
            TS_param_val = [np_p['TS'] for np_p in self.np_params]
            cnn_param_val = [np_p['Conv'] for np_p in self.np_params]
            fc_param_val = [np_p['FC'] for np_p in self.np_params]
        else:
            KB_param_val = get_value_of_valid_tensors(sess, self.shared_conv_KB_params)
            TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.conv_TS_params]
            #gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.dfcnn_gen_conv_params]
            cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.conv_params]
            fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        #parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['conv_trainable_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val

    def convert_tfVar_to_npVar(self, sess):
        if (self.num_tasks == 1 and self.task_is_new):
            orig_KB = [None for _ in range(self.num_conv_layers)]
        else:
            orig_KB = list(self.np_params[0]['KB'])

        def list_param_converter(list_of_params):
            converted_params = []
            for p in list_of_params:
                if type(p) == np.ndarray:
                    converted_params.append(p)
                elif _tf_tensor(p):
                    converted_params.append(sess.run(p))
                else:
                    converted_params.append(p)  ## append 'None' param
            return converted_params

        def double_list_param_converter(list_of_params):
            converted_params = []
            for task_params in list_of_params:
                converted_params.append(list_param_converter(task_params))
            return converted_params

        def post_process(layers_to_share, original_KB, updated_KB, updated_TS, updated_conv):
            for layer_cnt, (sharing_flag) in enumerate(layers_to_share):
                if sharing_flag:
                    ### Sharing this layer -> use new KB, TS and generated conv (no action needed), and make conv param None
                    updated_conv[self.current_task][2*layer_cnt], updated_conv[self.current_task][2*layer_cnt+1] = None, None
                else:
                    ### Not sharing this layer -> roll back KB, make TS and generated conv None, and keep conv param (no action needed)
                    updated_KB[layer_cnt] = original_KB[layer_cnt]
                    for tmptmp_cnt in range(5):
                        updated_TS[self.current_task][5*layer_cnt+tmptmp_cnt] = None
            return updated_KB, updated_TS, updated_conv

        self.np_params = []
        np_KB = list_param_converter(self.shared_conv_KB_params)
        np_TS = double_list_param_converter(self.conv_TS_params)
        np_conv = double_list_param_converter(self.conv_params)
        np_fc = double_list_param_converter(self.fc_params)
        if self.task_is_new:
            learned_config = self._possible_configs[self.best_config(sess)]
            self.conv_sharing.append(learned_config)
            np_KB, np_TS, np_conv = post_process(learned_config, orig_KB, np_KB, np_TS, np_conv)

        for cnt, (t, c, f) in enumerate(zip(np_TS, np_conv, np_fc)):
            self.np_params.append({'KB': np_KB, 'TS': t, 'Conv': c, 'FC': f} if cnt<1 else {'TS': t, 'Conv': c, 'FC': f})

    def _collect_trainable_variables(self):
        return_list = []
        for p in self.shared_conv_KB_params:
            if p is not None:
                return_list.append(p)
        for p in self.conv_TS_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.conv_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.fc_params[-1]:
            if p is not None:
                return_list.append(p)
        return return_list


##########################################################
#### Hybrid DF-CNN with auto sharing via EM algorithm ####
##########################################################
class LL_hybrid_DFCNN_EM_algo(Lifelong_Model_EM_Algo_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)
        self.dfcnn_KB_size = model_hyperpara['cnn_KB_sizes']
        self.dfcnn_TS_size = model_hyperpara['cnn_TS_sizes']
        self.dfcnn_stride_size = model_hyperpara['cnn_deconv_stride_sizes']
        self.dfcnn_KB_reg_scale = model_hyperpara['regularization_scale'][1]
        self.dfcnn_TS_reg_scale = model_hyperpara['regularization_scale'][3]

    def _build_task_model(self, net_input, output_size, task_cnt, params=None, trainable=False):
        if params is None:
            params_KB, params_TS, params_conv, params_fc = None, None, None, None
        else:
            params_KB, params_TS, params_conv, params_fc = params['KB'], params['TS'], params['Conv'], params['FC']

        if params_KB is not None:
            assert (len(params_KB) == self.num_conv_layers), "Given trained parameters of DF KB doesn't match to the hyper-parameters!"
        if params_TS is not None:
            assert (len(params_TS) == 4*self.num_conv_layers), "Given trained parameters of DF TS doesn't match to the hyper-parameters!"
        if params_conv is not None:
            assert (len(params_conv) == 2*self.num_conv_layers), "Given trained parameters of conv doesn't match to the hyper-parameters!"
        if params_fc is not None:
            assert (len(params_fc) == 2*self.num_fc_layers), "Given trained parameters of fc doesn't match to the hyper-parameters!"

        if (task_cnt==self.current_task) and self.task_is_new:
            ## New task model => use EM algorithm
            ### 1. make parameters used in any of configs
            with tf.name_scope('Hybrid_DFCNN_auto_EM'):
                dfcnn_TS_param_tmp, conv_params = [], []
                for layer_cnt in range(self.num_conv_layers):
                    dfcnn_TS_param_tmp += new_ELLA_cnn_deconv_tensordot_TS_param([[self.dfcnn_TS_size[2*layer_cnt], self.dfcnn_TS_size[2*layer_cnt], self.dfcnn_TS_size[2*layer_cnt+1], self.dfcnn_KB_size[2*layer_cnt+1]], [1, 1, 1, self.dfcnn_TS_size[2*layer_cnt+1]], [self.dfcnn_TS_size[2*layer_cnt+1], self.cnn_channels_size[layer_cnt], self.cnn_channels_size[layer_cnt+1]], [self.cnn_channels_size[layer_cnt+1]]], layer_cnt, task_cnt, self.TS_l2_reg, [None, None, None, None], trainable=trainable)
                    conv_params += [new_weight(shape=self.cnn_kernel_size[2*layer_cnt:2*(layer_cnt+1)]+self.cnn_channels_size[layer_cnt:layer_cnt+2], trainable=trainable, name='Conv_W%d'%(layer_cnt)), new_bias(shape=[self.cnn_channels_size[layer_cnt+1]], trainable=trainable, name='Conv_b%d'%(layer_cnt))]

                ### 2. generate task_models based on configs in self._possible_configs
                task_net, fc_params = [], None
                for conf in self._possible_configs:
                    net_tmp, _, _, _, _, _, fc_params = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], conf, self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=params_KB, cnn_TS_params=dfcnn_TS_param_tmp, cnn_params=conv_params, fc_activation_fn=self.hidden_act, fc_params=fc_params, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable)
                    task_net.append(net_tmp[-1])
        else:
            ## Hybrid DF-CNN with the learned configuration
            task_net, _, dfcnn_TS_param_tmp, gen_conv_param_tmp, conv_params, _, fc_params = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], self.conv_sharing[task_cnt], self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=params_KB, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable)
        self.dfcnn_TS_params.append(dfcnn_TS_param_tmp)
        #self.dfcnn_gen_conv_params.append(gen_conv_param_tmp)
        return task_net, conv_params, fc_params

    def _build_whole_model(self):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            if (task_cnt==self.current_task) and (self.task_is_new):
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': None, 'Conv': None, 'FC': None}
            else:
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': self.np_params[task_cnt]['TS'], 'Conv': self.np_params[task_cnt]['Conv'], 'FC': self.np_params[task_cnt]['FC']}
            task_net, conv_params, fc_params = self._build_task_model(x_b, num_classes, task_cnt, params=param_to_reuse, trainable=(task_cnt==self.current_task))

            self.task_models.append(task_net)
            self.conv_params.append(conv_params)
            self.fc_params.append(fc_params)
            self.params.append(self._collect_trainable_variables())
            self.num_trainable_var += count_trainable_var2(self.params[-1]) if task_cnt < 1 else count_trainable_var2(self.params[-1]) - self.dfcnn_KB_params_size

        self.dfcnn_KB_trainable_param = get_list_of_valid_tensors(self.dfcnn_KB_params)
        self.dfcnn_TS_trainable_param = get_list_of_valid_tensors(self.dfcnn_TS_params[self.current_task])
        self.conv_trainable_param = get_list_of_valid_tensors(self.conv_params[self.current_task])
        self.fc_trainable_param = get_list_of_valid_tensors(self.fc_params[self.current_task])
        self.trainable_params = list(self.dfcnn_KB_trainable_param) + list(self.dfcnn_TS_trainable_param) + list(self.conv_trainable_param) + list(self.fc_trainable_param)

    def _shared_param_init(self):
        self.dfcnn_TS_params, self.dfcnn_gen_conv_params = [], []
        self.KB_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_KB_reg_scale)
        self.TS_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_TS_reg_scale)
        KB_init_val = self.np_params[0]['KB'] if hasattr(self, 'np_params') else [None for _ in range(self.num_conv_layers)]
        self.dfcnn_KB_params = [new_ELLA_KB_param([1, self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt+1]], layer_cnt, 0, self.KB_l2_reg, KB_init_val[layer_cnt], True) for layer_cnt in range(self.num_conv_layers)]
        self.dfcnn_KB_params_size = count_trainable_var2(self.dfcnn_KB_params)

    def define_opt(self):
        with tf.name_scope('Optimization'):
            reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            KB_reg_term2 = tf.contrib.layers.apply_regularization(self.KB_l2_reg, reg_var)
            TS_reg_term2 = tf.contrib.layers.apply_regularization(self.TS_l2_reg, reg_var)

            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            if self.task_is_new:
                self.posterior_placeholder = tf.placeholder(tf.float32, shape=self.posterior.shape)
                self.update_prior = tf.assign(self.prior, self.posterior_placeholder)

                self.grads = [[] for _ in range(len(self.trainable_params))]
                for Loss in self.loss_for_train:
                    KB_grads = tf.gradients(Loss + KB_reg_term2, self.dfcnn_KB_trainable_param)
                    TS_grads = tf.gradients(Loss + TS_reg_term2, self.dfcnn_TS_trainable_param)
                    conv_grads = tf.gradients(Loss, self.conv_trainable_param)
                    fc_grads = tf.gradients(Loss, self.fc_trainable_param)

                    collected_grads = list(KB_grads) + list(TS_grads) + list(conv_grads) + list(fc_grads)
                    for param_index, (c_g) in enumerate(collected_grads):
                        self.grads[param_index].append(c_g)

                weighted_summed_grads = []
                for grad_list in self.grads:
                    weighted_summed_grads.append(self._weighted_sum_grads(grad_list, self.posterior_placeholder))
                grads_vars = [(grad, param) for grad, param in zip(weighted_summed_grads, self.trainable_params)]
                self.update = trainer.apply_gradients(grads_vars)
            else:
                KB_grads = tf.gradients(self.loss[self.current_task] + KB_reg_term2, self.dfcnn_KB_trainable_param)
                KB_grads_vars = [(grad, param) for grad, param in zip(KB_grads, self.dfcnn_KB_trainable_param)]

                TS_grads = tf.gradients(self.loss[self.current_task] + TS_reg_term2, self.dfcnn_TS_trainable_param)
                TS_grads_vars = [(grad, param) for grad, param in zip(TS_grads, self.dfcnn_TS_trainable_param)]

                conv_grads = tf.gradients(self.loss[self.current_task], self.conv_trainable_param)
                conv_grads_vars = [(grad, param) for grad, param in zip(conv_grads, self.conv_trainable_param)]

                fc_grads = tf.gradients(self.loss[self.current_task], self.fc_trainable_param)
                fc_grads_vars = [(grad, param) for grad, param in zip(fc_grads, self.fc_trainable_param)]

                self.grads = list(KB_grads) + list(TS_grads) + list(conv_grads) + list(fc_grads)
                grads_vars = KB_grads_vars + TS_grads_vars + conv_grads_vars + fc_grads_vars
                self.update = trainer.apply_gradients(grads_vars)

    def get_params_val(self, sess, use_npparams=True):
        if use_npparams:
            KB_param_val = self.np_params[0]['KB']
            TS_param_val = [np_p['TS'] for np_p in self.np_params]
            cnn_param_val = [np_p['Conv'] for np_p in self.np_params]
            fc_param_val = [np_p['FC'] for np_p in self.np_params]
        else:
            KB_param_val = get_value_of_valid_tensors(sess, self.dfcnn_KB_params)
            TS_param_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.dfcnn_TS_params]
            #gen_param_val = [get_value_of_valid_tensors(sess, cnn_gen_param) for cnn_gen_param in self.dfcnn_gen_conv_params]
            cnn_param_val = [get_value_of_valid_tensors(sess, cnn_param) for cnn_param in self.conv_params]
            fc_param_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        parameters_val['conv_KB'] = savemat_wrapper(KB_param_val)
        parameters_val['conv_TS'] = savemat_wrapper_nested_list(TS_param_val)
        #parameters_val['conv_generated_weights'] = savemat_wrapper_nested_list(gen_param_val)
        parameters_val['conv_trainable_weights'] = savemat_wrapper_nested_list(cnn_param_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_param_val)
        return parameters_val

    def convert_tfVar_to_npVar(self, sess):
        if (self.num_tasks == 1 and self.task_is_new):
            orig_KB = [None for _ in range(self.num_conv_layers)]
        else:
            orig_KB = list(self.np_params[0]['KB'])

        def list_param_converter(list_of_params):
            converted_params = []
            for p in list_of_params:
                if type(p) == np.ndarray:
                    converted_params.append(p)
                elif _tf_tensor(p):
                    converted_params.append(sess.run(p))
                else:
                    converted_params.append(p)  ## append 'None' param
            return converted_params

        def double_list_param_converter(list_of_params):
            converted_params = []
            for task_params in list_of_params:
                converted_params.append(list_param_converter(task_params))
            return converted_params

        def post_process(layers_to_share, original_KB, updated_KB, updated_TS, updated_conv):
            for layer_cnt, (sharing_flag) in enumerate(layers_to_share):
                if sharing_flag:
                    ### Sharing this layer -> use new KB, TS and generated conv (no action needed), and make conv param None
                    updated_conv[self.current_task][2*layer_cnt], updated_conv[self.current_task][2*layer_cnt+1] = None, None
                else:
                    ### Not sharing this layer -> roll back KB, make TS and generated conv None, and keep conv param (no action needed)
                    updated_KB[layer_cnt] = original_KB[layer_cnt]
                    for tmptmp_cnt in range(4):
                        updated_TS[self.current_task][4*layer_cnt+tmptmp_cnt] = None
            return updated_KB, updated_TS, updated_conv

        self.np_params = []
        np_KB = list_param_converter(self.dfcnn_KB_params)
        np_TS = double_list_param_converter(self.dfcnn_TS_params)
        np_conv = double_list_param_converter(self.conv_params)
        np_fc = double_list_param_converter(self.fc_params)
        if self.task_is_new:
            learned_config = self._possible_configs[self.best_config(sess)]
            self.conv_sharing.append(learned_config)
            np_KB, np_TS, np_conv = post_process(learned_config, orig_KB, np_KB, np_TS, np_conv)

        for cnt, (t, c, f) in enumerate(zip(np_TS, np_conv, np_fc)):
            self.np_params.append({'KB': np_KB, 'TS': t, 'Conv': c, 'FC': f} if cnt<1 else {'TS': t, 'Conv': c, 'FC': f})

    def _collect_trainable_variables(self):
        return_list = []
        for p in self.dfcnn_KB_params:
            if p is not None:
                return_list.append(p)
        for p in self.dfcnn_TS_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.conv_params[-1]:
            if p is not None:
                return_list.append(p)
        for p in self.fc_params[-1]:
            if p is not None:
                return_list.append(p)
        return return_list