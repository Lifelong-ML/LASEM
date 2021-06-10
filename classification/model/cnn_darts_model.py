import tensorflow as tf
import numpy as np
from random import shuffle

from utils.utils import get_value_of_valid_tensors, savemat_wrapper, savemat_wrapper_nested_list, count_trainable_var2
from utils.utils import new_weight, new_bias, new_ELLA_KB_param, get_list_of_valid_tensors, data_x_add_dummy, data_x_and_y_add_dummy
from utils.utils_nn import new_flexible_hardparam_cnn_fc_nets, new_darts_cnn_fc_net
from utils.utils_df_nn import new_ELLA_flexible_cnn_deconv_tensordot_fc_net, new_darts_dfcnn_fc_net
from classification.model.lifelong_model_frame import Lifelong_Model_Frame

_tf_ver = tf.__version__.split('.')
_up_to_date_tf = int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) >= 14)
if _up_to_date_tf:
    _tf_tensor = tf.is_tensor
else:
    _tf_tensor = tf.contrib.framework.is_tensor


########################################################
####   DARTS (Differentiable Architecture Search)   ####
####     based Selective Sharing baseline model     ####
########################################################
class LL_HPS_CNN_DARTS_net(Lifelong_Model_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)
        self.approx_order=model_hyperpara['darts_approx_order']
        self.conv_sharing = []

        def _possible_choices(input_subsets):
            list_subsets = []
            for c in [False, True]:
                for elem in input_subsets:
                    list_subsets.append(elem+[c])
            return list_subsets

        self._possible_configs = [[]]
        for layer_cnt in range(self.num_conv_layers):
            self._possible_configs = _possible_choices(self._possible_configs)
        self.num_possible_configs = len(self._possible_configs)

    def _build_task_model(self, net_input, output_size, task_cnt, params=None, trainable=False):
        if params is None:
            params_shared_conv, params_TS_conv, params_fc = None, None, None
        else:
            params_shared_conv, params_TS_conv, params_fc = params['Shared_Conv'], params['TS_Conv'], params['FC']

        if params_TS_conv is not None:
            assert (len(params_TS_conv) == 2*self.num_conv_layers), "Given trained parameters of conv doesn't match to the hyper-parameters!"
        if params_fc is not None:
            assert (len(params_fc) == 2*self.num_fc_layers), "Given trained parameters of fc doesn't match to the hyper-parameters!"

        eval_net = []
        if (task_cnt==self.current_task) and self.task_is_new:
            ## DARTS-based Hybrid HPS
            with tf.name_scope('DARTS_HPS'):
                task_net, _, conv_TS_params, conv_select_params, fc_params = new_darts_cnn_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], cnn_activation_fn=self.hidden_act, cnn_shared_params=params_shared_conv, cnn_TS_params=params_TS_conv, select_params=None, fc_activation_fn=self.hidden_act, fc_params=params_fc, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2], trainable=trainable)
                self.conv_select_params = conv_select_params

                ## build network for evaluation
                for conf in self._possible_configs:
                    net_tmp, _, _, _ = new_flexible_hardparam_cnn_fc_nets(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], conf, cnn_activation_fn=self.hidden_act, shared_cnn_params=params_shared_conv, cnn_params=conv_TS_params, fc_activation_fn=self.hidden_act, fc_params=fc_params, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, padding_type=self.padding_type, input_size=self.input_size[0:2], trainable=trainable, trainable_shared=trainable)
                    eval_net.append(net_tmp[-1])
        else:
            ## Hybrid HPS with the learned configuration
            task_net, conv_TS_params, _, fc_params = new_flexible_hardparam_cnn_fc_nets(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], self.conv_sharing[task_cnt], cnn_activation_fn=self.hidden_act, shared_cnn_params=params_shared_conv, cnn_params=params_TS_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, padding_type=self.padding_type, input_size=self.input_size[0:2], trainable=trainable, trainable_shared=trainable)
        return task_net, eval_net, conv_TS_params, fc_params

    def _build_whole_model(self):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            if (task_cnt==self.current_task) and (self.task_is_new):
                param_to_reuse = {'Shared_Conv': self.shared_conv_params, 'TS_Conv': None, 'FC': None}
            else:
                param_to_reuse = {'Shared_Conv': self.shared_conv_params, 'TS_Conv': self.np_params[task_cnt]['TS_Conv'], 'FC': self.np_params[task_cnt]['FC']}
            task_net, eval_net, conv_TS_params, fc_params = self._build_task_model(x_b, num_classes, task_cnt, params=param_to_reuse, trainable=(task_cnt==self.current_task))

            self.task_models.append(task_net)
            self.conv_params.append(conv_TS_params)
            self.fc_params.append(fc_params)
            self.params.append(self._collect_trainable_variables())
            self.num_trainable_var += count_trainable_var2(self.params[-1]) if task_cnt < 1 else count_trainable_var2(self.params[-1]) - self.shared_conv_params_size

            if len(eval_net) > 0:
                self.darts_eval_models = eval_net

        #self.conv_trainable_param = get_list_of_valid_tensors(self.conv_params[self.current_task])
        #self.fc_trainable_param = get_list_of_valid_tensors(self.fc_params[self.current_task])
        #self.trainable_params = list(self.dfcnn_KB_trainable_param) + list(self.dfcnn_TS_trainable_param) + list(self.conv_trainable_param) + list(self.fc_trainable_param)

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False):
        self.conv_select_params, self.darts_eval_models = None, None
        self._shared_param_init()
        super().add_new_task(output_dim, curr_task_index, single_input_placeholder=single_input_placeholder)

    def _shared_param_init(self):
        shared_conv_init_val = self.np_params[0]['Shared_Conv'] if hasattr(self, 'np_params') else [None for _ in range(2*self.num_conv_layers)]
        self.shared_conv_params = []
        for layer_cnt in range(self.num_conv_layers):
            self.shared_conv_params.append(new_weight(shape=self.cnn_kernel_size[2*layer_cnt:2*(layer_cnt+1)]+self.cnn_channels_size[layer_cnt:layer_cnt+2], init_tensor=shared_conv_init_val[2*layer_cnt], trainable=True, name='Shared_Conv_W%d'%(layer_cnt)))
            self.shared_conv_params.append(new_bias(shape=[self.cnn_channels_size[layer_cnt+1]], init_tensor=shared_conv_init_val[2*layer_cnt+1], trainable=True, name='Shared_Conv_b%d'%(layer_cnt)))
        self.shared_conv_params_size = count_trainable_var2(self.shared_conv_params)

    def get_darts_selection_val(self, sess):
        return get_value_of_valid_tensors(sess, self.conv_select_params)

    def get_params_val(self, sess, use_npparams=True):
        selection_params_val = self.get_darts_selection_val(sess)
        if use_npparams:
            shared_conv_val = self.np_params[0]['Shared_Conv']
            TS_conv_val = [np_p['TS_Conv'] for np_p in self.np_params]
            fc_val = [np_p['FC'] for np_p in self.np_params]
        else:
            shared_conv_val = get_value_of_valid_tensors(sess, self.shared_conv_params)
            TS_conv_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.conv_params]
            fc_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        parameters_val['DARTS_selection_param'] = savemat_wrapper(selection_params_val)
        parameters_val['shared_conv'] = savemat_wrapper(shared_conv_val)
        parameters_val['TS_conv'] = savemat_wrapper_nested_list(TS_conv_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_val)
        return parameters_val

    def best_config(self, sess):
        ## return the index of appropriate sharing configuration (self._possible_configs) according to the value of DARTS selection parameters
        selection_val = self.get_darts_selection_val(sess)
        # argmax 0 -> task-specific / argmax 1 -> shared
        selected_config_index = 0
        for layer_cnt, (layer_select) in enumerate(selection_val):
            selected_config_index = selected_config_index + np.argmax(layer_select) * (2**layer_cnt)
        return selected_config_index

    def darts_learned_selection(self, sess):
        ## return the list of decision (T:shared/F:task-specific) of sharing in each layer according to the value of DARTS selection parameters
        ## for elements of self.conv_sharing (e.g. 'bottom2' : [TTFFF..])
        selection_val = self.get_darts_selection_val(sess)
        sharing_flags = []
        for layer_select in selection_val:
            sharing_flags.append(np.argmax(layer_select))
        return sharing_flags

    def define_eval(self):
        with tf.name_scope('Model_Eval'):
            mask = tf.reshape(tf.cast(tf.range(self.batch_size)<self.num_data_in_batch, dtype=tf.float32), [self.batch_size, 1])
            self.eval = [tf.nn.softmax(task_model[-1])*mask for task_model in self.task_models]
            self.pred = [tf.argmax(task_model[-1]*mask, 1) for task_model in self.task_models]
            if self.task_is_new:
                self.eval_for_new_task = [tf.nn.softmax(task_model)*mask for task_model in self.darts_eval_models]
                self.pred_for_new_task = [tf.argmax(task_model*mask, 1) for task_model in self.darts_eval_models]

    def _loss_func(self, y1, y2):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y1, tf.int32), logits=y2))

    def define_loss(self):
        with tf.name_scope('Model_Loss'):
            self.loss = [self._loss_func(y_batch, task_model[-1]) for y_batch, task_model in zip(self.y_batch, self.task_models)]

    def define_accuracy(self):
        with tf.name_scope('Model_Accuracy'):
            mask = tf.cast(tf.range(self.batch_size)<self.num_data_in_batch, dtype=tf.float32)
            self.accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(task_model[-1], 1), tf.cast(y_batch, tf.int64)), tf.float32)*mask) for y_batch, task_model in zip(self.y_batch, self.task_models)]
            if self.task_is_new:
                self.accuracy_for_new_task = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(task_model, 1), tf.cast(self.y_batch[self.current_task], tf.int64)), tf.float32)*mask) for task_model in self.darts_eval_models]

    def define_opt(self):
        with tf.name_scope('Optimization'):
            self.grads = tf.gradients(self.loss[self.current_task], self.params[self.current_task])
            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            self.update = trainer.apply_gradients(list(zip(self.grads, self.params[self.current_task])))
            if self.task_is_new:
                if self.approx_order == 1:
                    self.selection_grads = tf.gradients(self.loss[self.current_task], self.conv_select_params)
                elif self.approx_order == 2:
                    #new_approx_params = [p-g*(self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)) for (p, g) in zip(self.params[self.current_task], self.grads)]
                    new_approx_params = [p-g*self.learn_rate for (p, g) in zip(self.params[self.current_task], self.grads)]
                    new_shared_conv = new_approx_params[0:2*self.num_conv_layers]
                    new_TS_conv = new_approx_params[2*self.num_conv_layers:4*self.num_conv_layers]
                    new_fc = new_approx_params[4*self.num_conv_layers:]

                    unrolled_model, _, _, _, _ = new_darts_cnn_fc_net(self.x_batch[self.current_task], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[self.output_sizes[self.current_task]], cnn_activation_fn=self.hidden_act, cnn_shared_params=new_shared_conv, cnn_TS_params=new_TS_conv, select_params=self.conv_select_params, fc_activation_fn=self.hidden_act, fc_params=new_fc, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2])
                    unrolled_loss = self._loss_func(self.y_batch[self.current_task], unrolled_model[-1])
                    #self.selection_grads = tf.gradients(unrolled_loss, self.conv_select_params)
                    selection_grads = tf.gradients(unrolled_loss, self.conv_select_params)
                    dw = tf.gradients(unrolled_loss, new_approx_params)

                    ## compute partial gradient approximating hessian
                    ratios = [0.01/tf.norm(g) for g in dw]
                    approx_params_upper = [p+g*r for (p, g, r) in zip(new_approx_params, dw, ratios)]
                    upper_model, _, _, _, _ = new_darts_cnn_fc_net(self.x_batch[self.current_task], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[self.output_sizes[self.current_task]], cnn_activation_fn=self.hidden_act, cnn_shared_params=approx_params_upper[0:2*self.num_conv_layers], cnn_TS_params=approx_params_upper[2*self.num_conv_layers:4*self.num_conv_layers], select_params=self.conv_select_params, fc_activation_fn=self.hidden_act, fc_params=approx_params_upper[4*self.num_conv_layers:], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2])
                    upper_loss = self._loss_func(self.y_batch[self.current_task], upper_model[-1])
                    upper_grad = tf.gradients(upper_loss, self.conv_select_params)

                    approx_params_lower = [p-g*r for (p, g, r) in zip(new_approx_params, dw, ratios)]
                    lower_model, _, _, _, _ = new_darts_cnn_fc_net(self.x_batch[self.current_task], self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[self.output_sizes[self.current_task]], cnn_activation_fn=self.hidden_act, cnn_shared_params=approx_params_lower[0:2*self.num_conv_layers], cnn_TS_params=approx_params_lower[2*self.num_conv_layers:4*self.num_conv_layers], select_params=self.conv_select_params, fc_activation_fn=self.hidden_act, fc_params=approx_params_lower[4*self.num_conv_layers:], padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, input_size=self.input_size[0:2])
                    lower_loss = self._loss_func(self.y_batch[self.current_task], lower_model[-1])
                    lower_grad = tf.gradients(lower_loss, self.conv_select_params)

                    #self.selection_grads = [g-(self.learn_rate/(1.0+self.epoch*self.learn_rate_decay)/(2*r))*(u-l) for (g, r, u, l) in zip(selection_grads, ratios, upper_grad, lower_grad)]
                    self.selection_grads = [g-(self.learn_rate/(2*r))*(u-l) for (g, r, u, l) in zip(selection_grads, ratios, upper_grad, lower_grad)]

                trainer2 = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
                self.selection_update = trainer2.apply_gradients(list(zip(self.selection_grads, self.conv_select_params)))

    def convert_tfVar_to_npVar(self, sess):
        if not (self.num_tasks == 1 and self.task_is_new):
            orig_KB = list(self.np_params[0]['Shared_Conv'])    ## copy of shared conv before training current task
        else:
            orig_KB = [None for _ in range(2*self.num_conv_layers)]

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

        def post_process(layers_to_share, original_KB, updated_KB, updated_conv):
            for layer_cnt, (sharing_flag) in enumerate(layers_to_share):
                if sharing_flag:
                    ### Sharing this layer -> use new KB, TS and generated conv (no action needed), and make conv param None
                    updated_conv[self.current_task][2*layer_cnt], updated_conv[self.current_task][2*layer_cnt+1] = None, None
                else:
                    ### Not sharing this layer -> roll back KB, make TS and generated conv None, and keep conv param (no action needed)
                    updated_KB[2*layer_cnt], updated_KB[2*layer_cnt+1] = original_KB[2*layer_cnt], original_KB[2*layer_cnt+1]
            return updated_KB, updated_conv

        self.np_params = []
        if len(self.conv_sharing) < self.num_tasks:
            self.conv_sharing.append(self.darts_learned_selection(sess))
        np_shared = list_param_converter(self.shared_conv_params)
        np_TS = double_list_param_converter(self.conv_params)
        np_fc = double_list_param_converter(self.fc_params)

        np_shared, np_TS = post_process(self.conv_sharing[self.current_task], orig_KB, np_shared, np_TS)
        for t, f in zip(np_TS, np_fc):
            self.np_params.append({'Shared_Conv': np_shared, 'TS_Conv': t, 'FC': f} if len(self.np_params)< 1 else {'TS_Conv': t, 'FC': f})

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

    def train_one_epoch(self, sess, data_x, data_y, epoch_cnt, task_index, learning_indices=None, augment_data=False, dropout_prob=1.0):
        task_model_index = self.find_task_model(task_index)
        num_train = data_x.shape[0]
        if learning_indices is None:
            learning_indices = list(range(num_train))
        shuffle(learning_indices)

        for batch_cnt in range(num_train//self.batch_size):
            batch_train_x = data_x[learning_indices[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size]]
            batch_train_y = data_y[learning_indices[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size]]

            if self.task_is_new:
                ## Update architecture (selection param)
                sess.run(self.selection_update, feed_dict={self.model_input[task_model_index]: batch_train_x, self.true_output[task_model_index]: batch_train_y, self.epoch: epoch_cnt, self.dropout_prob: dropout_prob})

            ## Update NN weights
            sess.run(self.update, feed_dict={self.model_input[task_model_index]: batch_train_x, self.true_output[task_model_index]: batch_train_y, self.epoch: epoch_cnt, self.dropout_prob: dropout_prob})

    def eval_one_task(self, sess, data_x, task_index, dropout_prob=1.0):
        task_model_index = self.find_task_model(task_index)
        num_data, num_classes = data_x.shape[0], self.output_sizes[task_model_index]
        eval_output = np.zeros([num_data, num_classes], dtype=np.float32)

        num_batch = num_data//self.batch_size
        num_remains = num_data - self.batch_size*num_batch

        if self.task_is_new and (self.current_task == task_model_index):
            best_config = self.best_config(sess)
            eval_func = self.eval_for_new_task[best_config]
        else:
            eval_func = self.eval[task_model_index]

        for batch_cnt in range(num_batch):
            eval_output[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size] = sess.run(eval_func, feed_dict={self.model_input: data_x[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.dropout_prob: dropout_prob, self.num_data_in_batch: self.batch_size})
        if num_remains > 0:
            temp_pred = sess.run(eval_func, feed_dict={self.model_input: data_x_add_dummy(data_x[-num_remains:], self.batch_size), self.dropout_prob: dropout_prob, self.num_data_in_batch: num_remains})
            eval_output[-num_remains:] = temp_pred[0:num_remains]
        return eval_output

    def infer_one_task(self, sess, data_x, task_index, dropout_prob=1.0):
        task_model_index = self.find_task_model(task_index)
        num_data = data_x.shape[0]
        inferred_labels = np.zeros(num_data, dtype=np.int32)

        num_batch = num_data//self.batch_size
        num_remains = num_data - self.batch_size*num_batch

        if self.task_is_new and (self.current_task == task_model_index):
            best_config = self.best_config(sess)
            pred_func = self.pred_for_new_task[best_config]
        else:
            pred_func = self.pred[task_model_index]

        for batch_cnt in range(num_batch):
            temp_pred = sess.run(pred_func, feed_dict={self.model_input[task_model_index]: data_x[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.dropout_prob: dropout_prob, self.num_data_in_batch: self.batch_size})
            inferred_labels[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size] = np.squeeze(temp_pred)
        if num_remains > 0:
            temp_pred = sess.run(pred_func, feed_dict={self.model_input[task_model_index]: data_x_add_dummy(data_x[-num_remains:], self.batch_size), self.dropout_prob: dropout_prob, self.num_data_in_batch: num_remains})
            inferred_labels[-num_remains:] = np.squeeze(temp_pred[0:num_remains])
        return inferred_labels

    def compute_accuracy_one_task(self, sess, data_x, data_y, task_index, dropout_prob=1.0):
        task_model_index = self.find_task_model(task_index)
        num_data, accuracy = data_x.shape[0], 0.0

        num_batch = num_data//self.batch_size
        num_remains = num_data - self.batch_size*num_batch

        if self.task_is_new and (self.current_task == task_model_index):
            best_config = self.best_config(sess)
            acc_func = self.accuracy_for_new_task[best_config]
        else:
            acc_func = self.accuracy[task_model_index]

        for batch_cnt in range(num_batch):
            accuracy += sess.run(acc_func, feed_dict={self.model_input[task_model_index]: data_x[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.true_output[task_model_index]: data_y[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.dropout_prob: dropout_prob, self.num_data_in_batch: self.batch_size})
        if num_remains > 0:
            tmp_x, tmp_y = data_x_and_y_add_dummy(data_x[-num_remains:], data_y[-num_remains:], self.batch_size)
            accuracy += sess.run(acc_func, feed_dict={self.model_input[task_model_index]: tmp_x, self.true_output[task_model_index]: tmp_y, self.dropout_prob: dropout_prob, self.num_data_in_batch: num_remains})
        return float(accuracy)/float(num_data)


########################################################
####   DARTS (Differentiable Architecture Search)   ####
####     based Selective Sharing baseline model     ####
########################################################
class LL_DFCNN_DARTS_net(Lifelong_Model_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)
        self.dfcnn_KB_size = model_hyperpara['cnn_KB_sizes']
        self.dfcnn_TS_size = model_hyperpara['cnn_TS_sizes']
        self.dfcnn_stride_size = model_hyperpara['cnn_deconv_stride_sizes']
        self.dfcnn_KB_reg_scale = model_hyperpara['regularization_scale'][1]
        self.dfcnn_TS_reg_scale = model_hyperpara['regularization_scale'][3]
        self.approx_order=model_hyperpara['darts_approx_order']
        self.conv_sharing = []

        def _possible_choices(input_subsets):
            list_subsets = []
            for c in [False, True]:
                for elem in input_subsets:
                    list_subsets.append(elem+[c])
            return list_subsets

        self._possible_configs = [[]]
        for layer_cnt in range(self.num_conv_layers):
            self._possible_configs = _possible_choices(self._possible_configs)
        self.num_possible_configs = len(self._possible_configs)

    def _build_task_model(self, net_input, output_size, task_cnt, params=None, trainable=False):
        if params is None:
            params_KB, params_TS, params_conv, params_fc = None, None, None, None
        else:
            params_KB, params_TS, params_conv, params_fc = params['KB'], params['TS'], params['TS_Conv'], params['FC']

        if params_conv is not None:
            assert (len(params_conv) == 2*self.num_conv_layers), "Given trained parameters of conv doesn't match to the hyper-parameters!"
        if params_fc is not None:
            assert (len(params_fc) == 2*self.num_fc_layers), "Given trained parameters of fc doesn't match to the hyper-parameters!"

        eval_net = []
        if (task_cnt==self.current_task) and self.task_is_new:
            ## DARTS-based DF-CNN
            with tf.name_scope('DARTS_DFCNN'):
                task_net, _, dfcnn_TS_params, conv_params, conv_select_params, fc_params = new_darts_dfcnn_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, cnn_activation_fn=self.hidden_act, dfcnn_TS_activation_fn=None, fc_activation_fn=self.hidden_act, dfcnn_KB_params=params_KB, dfcnn_TS_params=params_TS, cnn_TS_params=params_conv, select_params=None, fc_params=params_fc, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, trainable=trainable, task_index=task_cnt)
                self.conv_select_params = conv_select_params

                ## build network for evaluation
                for conf in self._possible_configs:
                    net_tmp, _, _, _, _, _, _ = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], conf, self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=params_KB, cnn_TS_params=dfcnn_TS_params, cnn_params=conv_params, fc_activation_fn=self.hidden_act, fc_params=fc_params, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable)
                    eval_net.append(net_tmp[-1])
        else:
            ## DF-CNN with the learned configuration
            task_net, _, dfcnn_TS_params, _, conv_params, _, fc_params = new_ELLA_flexible_cnn_deconv_tensordot_fc_net(net_input, self.cnn_kernel_size, self.cnn_channels_size, self.cnn_stride_size, list(self.fc_size)+[output_size], self.conv_sharing[task_cnt], self.dfcnn_KB_size, self.dfcnn_TS_size, self.dfcnn_stride_size, cnn_activation_fn=self.hidden_act, cnn_para_activation_fn=None, cnn_KB_params=params_KB, cnn_TS_params=params_TS, cnn_params=params_conv, fc_activation_fn=self.hidden_act, fc_params=params_fc, KB_reg_type=self.KB_l2_reg, TS_reg_type=self.TS_l2_reg, padding_type=self.padding_type, max_pool=self.max_pooling, pool_sizes=self.pool_size, dropout=self.dropout, dropout_prob=self.dropout_prob, task_index=task_cnt, skip_connections=list(self.skip_connect), trainable=trainable)
        return task_net, eval_net, dfcnn_TS_params, conv_params, fc_params

    def _build_whole_model(self):
        for task_cnt, (num_classes, x_b) in enumerate(zip(self.output_sizes, self.x_batch)):
            if (task_cnt==self.current_task) and (self.task_is_new):
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': None, 'TS_Conv': None, 'FC': None}
            else:
                param_to_reuse = {'KB': self.dfcnn_KB_params, 'TS': self.np_params[task_cnt]['TS'], 'TS_Conv': self.np_params[task_cnt]['TS_Conv'], 'FC': self.np_params[task_cnt]['FC']}
            task_net, eval_net, dfcnn_TS_params, conv_TS_params, fc_params = self._build_task_model(x_b, num_classes, task_cnt, params=param_to_reuse, trainable=(task_cnt==self.current_task))

            self.task_models.append(task_net)
            self.dfcnn_TS_params.append(dfcnn_TS_params)
            self.conv_params.append(conv_TS_params)
            self.fc_params.append(fc_params)
            self.params.append(self._collect_trainable_variables())
            self.num_trainable_var += count_trainable_var2(self.params[-1]) if task_cnt < 1 else count_trainable_var2(self.params[-1]) - self.dfcnn_KB_params_size

            if len(eval_net) > 0:
                self.darts_eval_models = eval_net

        self.dfcnn_KB_trainable_param = get_list_of_valid_tensors(self.dfcnn_KB_params)
        self.dfcnn_TS_trainable_param = get_list_of_valid_tensors(self.dfcnn_TS_params[self.current_task])
        self.conv_trainable_param = get_list_of_valid_tensors(self.conv_params[self.current_task])
        self.fc_trainable_param = get_list_of_valid_tensors(self.fc_params[self.current_task])
        self.trainable_params = list(self.dfcnn_KB_trainable_param) + list(self.dfcnn_TS_trainable_param) + list(self.conv_trainable_param) + list(self.fc_trainable_param)

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False):
        self.conv_select_params, self.darts_eval_models = None, None
        self._shared_param_init()
        super().add_new_task(output_dim, curr_task_index, single_input_placeholder=single_input_placeholder)

    def _shared_param_init(self):
        self.dfcnn_TS_params = []
        self.KB_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_KB_reg_scale)
        self.TS_l2_reg = tf.contrib.layers.l2_regularizer(scale=self.dfcnn_TS_reg_scale)
        KB_init_val = self.np_params[0]['KB'] if hasattr(self, 'np_params') else [None for _ in range(self.num_conv_layers)]
        self.dfcnn_KB_params = [new_ELLA_KB_param([1, self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt], self.dfcnn_KB_size[2*layer_cnt+1]], layer_cnt, 0, self.KB_l2_reg, KB_init_val[layer_cnt], True) for layer_cnt in range(self.num_conv_layers)]
        self.dfcnn_KB_params_size = count_trainable_var2(self.dfcnn_KB_params)

    def get_darts_selection_val(self, sess):
        return get_value_of_valid_tensors(sess, self.conv_select_params)

    def get_params_val(self, sess, use_npparams=True):
        selection_params_val = self.get_darts_selection_val(sess)
        if use_npparams:
            KB_val = self.np_params[0]['KB']
            TS_val = [np_p['TS'] for np_p in self.np_params]
            TS_conv_val = [np_p['TS_Conv'] for np_p in self.np_params]
            fc_val = [np_p['FC'] for np_p in self.np_params]
        else:
            KB_val = get_value_of_valid_tensors(sess, self.dfcnn_KB_params)
            TS_val = [get_value_of_valid_tensors(sess, dfcnn_TS_param) for dfcnn_TS_param in self.dfcnn_TS_params]
            TS_conv_val = [get_value_of_valid_tensors(sess, cnn_TS_param) for cnn_TS_param in self.conv_params]
            fc_val = [get_value_of_valid_tensors(sess, fc_param) for fc_param in self.fc_params]

        parameters_val = {}
        parameters_val['DARTS_selection_param'] = savemat_wrapper(selection_params_val)
        parameters_val['KB'] = savemat_wrapper(KB_val)
        parameters_val['TS'] = savemat_wrapper_nested_list(TS_val)
        parameters_val['TS_conv'] = savemat_wrapper_nested_list(TS_conv_val)
        parameters_val['fc_weights'] = savemat_wrapper_nested_list(fc_val)
        return parameters_val

    def best_config(self, sess):
        ## return the index of appropriate sharing configuration (self._possible_configs) according to the value of DARTS selection parameters
        selection_val = self.get_darts_selection_val(sess)
        # argmax 0 -> task-specific / argmax 1 -> shared
        selected_config_index = 0
        for layer_cnt, (layer_select) in enumerate(selection_val):
            selected_config_index = selected_config_index + np.argmax(layer_select) * (2**layer_cnt)
        return selected_config_index

    def darts_learned_selection(self, sess):
        ## return the list of decision (T:shared/F:task-specific) of sharing in each layer according to the value of DARTS selection parameters
        ## for elements of self.conv_sharing (e.g. 'bottom2' : [TTFFF..])
        selection_val = self.get_darts_selection_val(sess)
        sharing_flags = []
        for layer_select in selection_val:
            sharing_flags.append(np.argmax(layer_select))
        return sharing_flags

    def define_eval(self):
        with tf.name_scope('Model_Eval'):
            mask = tf.reshape(tf.cast(tf.range(self.batch_size)<self.num_data_in_batch, dtype=tf.float32), [self.batch_size, 1])
            self.eval = [tf.nn.softmax(task_model[-1])*mask for task_model in self.task_models]
            self.pred = [tf.argmax(task_model[-1]*mask, 1) for task_model in self.task_models]
            if self.task_is_new:
                self.eval_for_new_task = [tf.nn.softmax(task_model)*mask for task_model in self.darts_eval_models]
                self.pred_for_new_task = [tf.argmax(task_model*mask, 1) for task_model in self.darts_eval_models]

    def _loss_func(self, y1, y2):
        return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y1, tf.int32), logits=y2))

    def define_loss(self):
        with tf.name_scope('Model_Loss'):
            self.loss = [self._loss_func(y_batch, task_model[-1]) for y_batch, task_model in zip(self.y_batch, self.task_models)]

    def define_accuracy(self):
        with tf.name_scope('Model_Accuracy'):
            mask = tf.cast(tf.range(self.batch_size)<self.num_data_in_batch, dtype=tf.float32)
            self.accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(task_model[-1], 1), tf.cast(y_batch, tf.int64)), tf.float32)*mask) for y_batch, task_model in zip(self.y_batch, self.task_models)]
            if self.task_is_new:
                self.accuracy_for_new_task = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(task_model, 1), tf.cast(self.y_batch[self.current_task], tf.int64)), tf.float32)*mask) for task_model in self.darts_eval_models]

    def define_opt(self):
        with tf.name_scope('Optimization'):
            reg_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            KB_reg_term2 = tf.contrib.layers.apply_regularization(self.KB_l2_reg, reg_var)
            TS_reg_term2 = tf.contrib.layers.apply_regularization(self.TS_l2_reg, reg_var)

            KB_grads = tf.gradients(self.loss[self.current_task] + KB_reg_term2, self.dfcnn_KB_trainable_param)
            KB_grads_vars = [(grad, param) for grad, param in zip(KB_grads, self.dfcnn_KB_trainable_param)]

            TS_grads = tf.gradients(self.loss[self.current_task] + TS_reg_term2, self.dfcnn_TS_trainable_param)
            TS_grads_vars = [(grad, param) for grad, param in zip(TS_grads, self.dfcnn_TS_trainable_param)]

            conv_grads = tf.gradients(self.loss[self.current_task], self.conv_trainable_param)
            conv_grads_vars = [(grad, param) for grad, param in zip(conv_grads, self.conv_trainable_param)]

            fc_grads = tf.gradients(self.loss[self.current_task], self.fc_trainable_param)
            fc_grads_vars = [(grad, param) for grad, param in zip(fc_grads, self.fc_trainable_param)]

            self.grads = list(KB_grads) + list(TS_grads) + list(conv_grads) + list(fc_grads)
            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            self.update = trainer.apply_gradients(KB_grads_vars + TS_grads_vars + conv_grads_vars + fc_grads_vars)
            if self.task_is_new:
                if self.approx_order == 1:
                    self.selection_grads = tf.gradients(self.loss[self.current_task], self.conv_select_params)
                elif self.approx_order == 2:
                    raise NotImplementedError("Not Implemented because of 2nd derivative Issue!")

                trainer2 = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
                self.selection_update = trainer2.apply_gradients(list(zip(self.selection_grads, self.conv_select_params)))

    def convert_tfVar_to_npVar(self, sess):
        if not (self.num_tasks == 1 and self.task_is_new):
            orig_KB = list(self.np_params[0]['KB'])    ## copy of shared conv before training current task
        else:
            orig_KB = [None for _ in range(2*self.num_conv_layers)]

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
                    ### Sharing this layer -> use new KB, TS, and make conv param None
                    updated_conv[self.current_task][2*layer_cnt], updated_conv[self.current_task][2*layer_cnt+1] = None, None
                else:
                    ### Not sharing this layer -> roll back KB, make TS None, and keep conv param (no action needed)
                    updated_KB[layer_cnt] = original_KB[layer_cnt]
                    updated_TS[self.current_task][4*layer_cnt], updated_TS[self.current_task][4*layer_cnt+1] = None, None
                    updated_TS[self.current_task][4*layer_cnt+2], updated_TS[self.current_task][4*layer_cnt+3] = None, None
            return updated_KB, updated_TS, updated_conv

        self.np_params = []
        if len(self.conv_sharing) < self.num_tasks:
            self.conv_sharing.append(self.darts_learned_selection(sess))
        np_KB = list_param_converter(self.dfcnn_KB_params)
        np_TS = double_list_param_converter(self.dfcnn_TS_params)
        np_conv = double_list_param_converter(self.conv_params)
        np_fc = double_list_param_converter(self.fc_params)

        np_KB, np_TS, np_conv = post_process(self.conv_sharing[self.current_task], orig_KB, np_KB, np_TS, np_conv)
        for t, c, f in zip(np_TS, np_conv, np_fc):
            self.np_params.append({'KB': np_KB, 'TS': t, 'TS_Conv': c, 'FC': f} if len(self.np_params)< 1 else {'TS': t, 'TS_Conv': c, 'FC': f})

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

    def train_one_epoch(self, sess, data_x, data_y, epoch_cnt, task_index, learning_indices=None, augment_data=False, dropout_prob=1.0):
        task_model_index = self.find_task_model(task_index)
        num_train = data_x.shape[0]
        if learning_indices is None:
            learning_indices = list(range(num_train))
        shuffle(learning_indices)

        for batch_cnt in range(num_train//self.batch_size):
            batch_train_x = data_x[learning_indices[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size]]
            batch_train_y = data_y[learning_indices[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size]]

            if self.task_is_new:
                ## Update architecture (selection param)
                sess.run(self.selection_update, feed_dict={self.model_input[task_model_index]: batch_train_x, self.true_output[task_model_index]: batch_train_y, self.epoch: epoch_cnt, self.dropout_prob: dropout_prob})

            ## Update NN weights
            sess.run(self.update, feed_dict={self.model_input[task_model_index]: batch_train_x, self.true_output[task_model_index]: batch_train_y, self.epoch: epoch_cnt, self.dropout_prob: dropout_prob})

    def eval_one_task(self, sess, data_x, task_index, dropout_prob=1.0):
        task_model_index = self.find_task_model(task_index)
        num_data, num_classes = data_x.shape[0], self.output_sizes[task_model_index]
        eval_output = np.zeros([num_data, num_classes], dtype=np.float32)

        num_batch = num_data//self.batch_size
        num_remains = num_data - self.batch_size*num_batch

        if self.task_is_new and (self.current_task == task_model_index):
            best_config = self.best_config(sess)
            eval_func = self.eval_for_new_task[best_config]
        else:
            eval_func = self.eval[task_model_index]

        for batch_cnt in range(num_batch):
            eval_output[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size] = sess.run(eval_func, feed_dict={self.model_input: data_x[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.dropout_prob: dropout_prob, self.num_data_in_batch: self.batch_size})
        if num_remains > 0:
            temp_pred = sess.run(eval_func, feed_dict={self.model_input: data_x_add_dummy(data_x[-num_remains:], self.batch_size), self.dropout_prob: dropout_prob, self.num_data_in_batch: num_remains})
            eval_output[-num_remains:] = temp_pred[0:num_remains]
        return eval_output

    def infer_one_task(self, sess, data_x, task_index, dropout_prob=1.0):
        task_model_index = self.find_task_model(task_index)
        num_data = data_x.shape[0]
        inferred_labels = np.zeros(num_data, dtype=np.int32)

        num_batch = num_data//self.batch_size
        num_remains = num_data - self.batch_size*num_batch

        if self.task_is_new and (self.current_task == task_model_index):
            best_config = self.best_config(sess)
            pred_func = self.pred_for_new_task[best_config]
        else:
            pred_func = self.pred[task_model_index]

        for batch_cnt in range(num_batch):
            temp_pred = sess.run(pred_func, feed_dict={self.model_input[task_model_index]: data_x[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.dropout_prob: dropout_prob, self.num_data_in_batch: self.batch_size})
            inferred_labels[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size] = np.squeeze(temp_pred)
        if num_remains > 0:
            temp_pred = sess.run(pred_func, feed_dict={self.model_input[task_model_index]: data_x_add_dummy(data_x[-num_remains:], self.batch_size), self.dropout_prob: dropout_prob, self.num_data_in_batch: num_remains})
            inferred_labels[-num_remains:] = np.squeeze(temp_pred[0:num_remains])
        return inferred_labels

    def compute_accuracy_one_task(self, sess, data_x, data_y, task_index, dropout_prob=1.0):
        task_model_index = self.find_task_model(task_index)
        num_data, accuracy = data_x.shape[0], 0.0

        num_batch = num_data//self.batch_size
        num_remains = num_data - self.batch_size*num_batch

        if self.task_is_new and (self.current_task == task_model_index):
            best_config = self.best_config(sess)
            acc_func = self.accuracy_for_new_task[best_config]
        else:
            acc_func = self.accuracy[task_model_index]

        for batch_cnt in range(num_batch):
            accuracy += sess.run(acc_func, feed_dict={self.model_input[task_model_index]: data_x[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.true_output[task_model_index]: data_y[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.dropout_prob: dropout_prob, self.num_data_in_batch: self.batch_size})
        if num_remains > 0:
            tmp_x, tmp_y = data_x_and_y_add_dummy(data_x[-num_remains:], data_y[-num_remains:], self.batch_size)
            accuracy += sess.run(acc_func, feed_dict={self.model_input[task_model_index]: tmp_x, self.true_output[task_model_index]: tmp_y, self.dropout_prob: dropout_prob, self.num_data_in_batch: num_remains})
        return float(accuracy)/float(num_data)