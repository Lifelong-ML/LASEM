import sys
from abc import abstractmethod
import numpy as np
import tensorflow as tf
from random import shuffle
from scipy.io import savemat, loadmat
from os import mkdir
import shutil
from copy import deepcopy

from utils.utils import data_augmentation_in_minibatch, data_x_add_dummy, data_x_and_y_add_dummy

_tf_ver = tf.__version__.split('.')
_up_to_date_tf = int(_tf_ver[0]) > 1 or (int(_tf_ver[0])==1 and int(_tf_ver[1]) > 14)
if _up_to_date_tf:
    _tf_tensor = tf.is_tensor
else:
    _tf_tensor = tf.contrib.framework.is_tensor

if sys.version_info.major < 3:
    from utils import count_trainable_var
else:
    from utils.utils import count_trainable_var

_debug_mode = True


class Lifelong_Model_Frame():
    def __init__(self, model_hyperpara, train_hyperpara):
        self.input_size = model_hyperpara['image_dimension']    ## img_width * img_height * img_channel
        self.cnn_channels_size = [self.input_size[-1]]+list(model_hyperpara['channel_sizes'])    ## include dim of input channel
        self.cnn_kernel_size = model_hyperpara['kernel_sizes']     ## len(kernel_size) == 2*(len(channels_size)-1)
        self.cnn_stride_size = model_hyperpara['stride_sizes']
        self.pool_size = model_hyperpara['pooling_size']      ## len(pool_size) == 2*(len(channels_size)-1)
        self.fc_size = model_hyperpara['hidden_layer']
        self.padding_type = model_hyperpara['padding_type']
        self.max_pooling = model_hyperpara['max_pooling']
        self.dropout = model_hyperpara['dropout']
        self.skip_connect = model_hyperpara['skip_connect']

        self.learn_rate = train_hyperpara['lr']
        self.learn_rate_decay = train_hyperpara['lr_decay']
        self.batch_size = model_hyperpara['batch_size']
        self.hidden_act = model_hyperpara['hidden_activation']

        self.num_conv_layers, self.num_fc_layers = len(self.cnn_channels_size)-1, len(self.fc_size)+1

        self.num_tasks = 0
        self.num_trainable_var = 0
        self.output_sizes = []
        self.task_indices = []

    @abstractmethod
    def _build_whole_model(self):
        raise NotImplementedError

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False):
        if self.is_new_task(curr_task_index):
            self.num_tasks += 1
            self.output_sizes.append(output_dim)
            self.task_indices.append(curr_task_index)
            self.task_is_new = True
        else:
            self.task_is_new = False
        self.num_trainable_var = 0
        self.current_task = self.find_task_model(curr_task_index)

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)
        self.num_data_in_batch = tf.placeholder(dtype=tf.int32)
        if single_input_placeholder:
            self.model_input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]])
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]

        with tf.name_scope('Minibatch_Data'):
            if single_input_placeholder:
                self.x_batch = [tf.reshape(self.model_input, [-1]+list(self.input_size)) for _ in range(self.num_tasks)]
            else:
                self.x_batch = [tf.reshape(x, [-1]+list(self.input_size)) for x in self.model_input]
            self.y_batch = [y for y in self.true_output]

        self.task_models, self.conv_params, self.fc_params, self.params = [], [], [], []
        self._build_whole_model()
        self.define_eval()
        self.define_loss()
        self.define_accuracy()
        self.define_opt()

    def is_new_task(self, curr_task_index):
        return (not (curr_task_index in self.task_indices))

    def find_task_model(self, task_index_to_search):
        return self.task_indices.index(task_index_to_search)

    def number_of_learned_tasks(self):
        return self.num_tasks

    def define_eval(self):
        with tf.name_scope('Model_Eval'):
            mask = tf.reshape(tf.cast(tf.range(self.batch_size)<self.num_data_in_batch, dtype=tf.float32), [self.batch_size, 1])
            self.eval = [tf.nn.softmax(task_model[-1])*mask for task_model in self.task_models]
            self.pred = [tf.argmax(task_model[-1]*mask, 1) for task_model in self.task_models]

    def define_loss(self):
        with tf.name_scope('Model_Loss'):
            self.loss = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batch, tf.int32), logits=task_model[-1])) for y_batch, task_model in zip(self.y_batch, self.task_models)]

    def define_accuracy(self):
        with tf.name_scope('Model_Accuracy'):
            mask = tf.cast(tf.range(self.batch_size)<self.num_data_in_batch, dtype=tf.float32)
            self.accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(task_model[-1], 1), tf.cast(y_batch, tf.int64)), tf.float32)*mask) for y_batch, task_model in zip(self.y_batch, self.task_models)]

    def define_opt(self):
        with tf.name_scope('Optimization'):
            self.grads = tf.gradients(self.loss[self.current_task], self.params[self.current_task])
            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            self.update = trainer.apply_gradients(list(zip(self.grads, self.params[self.current_task])))

    def convert_tfVar_to_npVar(self, sess):
        if len(self.params) > 0:
            converted_params = []
            for task_params in self.params:
                converted_task_params = []
                for p in task_params:
                    if type(p) == np.ndarray:
                        converted_task_params.append(p)
                    elif _tf_tensor(p):
                        converted_task_params.append(sess.run(p))
                    else:
                        print("\nData type of variable is not based on either TensorFlow or Numpy!!\n")
                        raise ValueError
                converted_params.append(converted_task_params)
            self.np_params = converted_params

    @abstractmethod
    def get_param(self, sess):
        raise NotImplementedError

    def train_one_epoch(self, sess, data_x, data_y, epoch_cnt, task_index, learning_indices=None, augment_data=False, dropout_prob=1.0):
        task_model_index = self.find_task_model(task_index)
        num_train = data_x.shape[0]
        if learning_indices is None:
            learning_indices = list(range(num_train))
        shuffle(learning_indices)

        for batch_cnt in range(num_train//self.batch_size):
            batch_train_x = data_x[learning_indices[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size]]
            batch_train_y = data_y[learning_indices[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size]]

            if augment_data:
                batch_train_x, batch_train_y = data_augmentation_in_minibatch(batch_train_x, batch_train_y, self.input_size)

            sess.run(self.update, feed_dict={self.model_input: batch_train_x, self.true_output[task_model_index]: batch_train_y, self.epoch: epoch_cnt, self.dropout_prob: dropout_prob})

    def eval_one_task(self, sess, data_x, task_index, dropout_prob=1.0):
        task_model_index = self.find_task_model(task_index)
        num_data, num_classes = data_x.shape[0], self.output_sizes[task_model_index]
        eval_output = np.zeros([num_data, num_classes], dtype=np.float32)

        num_batch = num_data//self.batch_size
        num_remains = num_data - self.batch_size*num_batch

        for batch_cnt in range(num_batch):
            eval_output[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size] = sess.run(self.eval[task_model_index], feed_dict={self.model_input: data_x[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.dropout_prob: dropout_prob, self.num_data_in_batch: self.batch_size})
        if num_remains > 0:
            temp_pred = sess.run(self.eval[task_model_index], feed_dict={self.model_input: data_x_add_dummy(data_x[-num_remains:], self.batch_size), self.dropout_prob: dropout_prob, self.num_data_in_batch: num_remains})
            eval_output[-num_remains:] = temp_pred[0:num_remains]
        return eval_output

    def infer_one_task(self, sess, data_x, task_index, dropout_prob=1.0):
        task_model_index = self.find_task_model(task_index)
        num_data = data_x.shape[0]
        inferred_labels = np.zeros(num_data, dtype=np.int32)

        num_batch = num_data//self.batch_size
        num_remains = num_data - self.batch_size*num_batch

        for batch_cnt in range(num_batch):
            temp_pred = sess.run(self.pred[task_model_index], feed_dict={self.model_input: data_x[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.dropout_prob: dropout_prob, self.num_data_in_batch: self.batch_size})
            inferred_labels[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size] = np.squeeze(temp_pred)
        if num_remains > 0:
            temp_pred = sess.run(self.pred[task_model_index], feed_dict={self.model_input: data_x_add_dummy(data_x[-num_remains:], self.batch_size), self.dropout_prob: dropout_prob, self.num_data_in_batch: num_remains})
            inferred_labels[-num_remains:] = np.squeeze(temp_pred[0:num_remains])
        return inferred_labels

    def compute_accuracy_one_task(self, sess, data_x, data_y, task_index, dropout_prob=1.0):
        task_model_index = self.find_task_model(task_index)
        num_data, accuracy = data_x.shape[0], 0.0

        num_batch = num_data//self.batch_size
        num_remains = num_data - self.batch_size*num_batch

        for batch_cnt in range(num_batch):
            accuracy += sess.run(self.accuracy[task_model_index], feed_dict={self.model_input: data_x[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.true_output[task_model_index]: data_y[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.dropout_prob: dropout_prob, self.num_data_in_batch: self.batch_size})
        if num_remains > 0:
            tmp_x, tmp_y = data_x_and_y_add_dummy(data_x[-num_remains:], data_y[-num_remains:], self.batch_size)
            accuracy += sess.run(self.accuracy[task_model_index], feed_dict={self.model_input: tmp_x, self.true_output[task_model_index]: tmp_y, self.dropout_prob: dropout_prob, self.num_data_in_batch: num_remains})
        return float(accuracy)/float(num_data)

    def compute_transferability_score_one_task(self, sess, data_x, data_y, comp_task_index, dropout_prob=1.0):
        task_model_index = self.find_task_model(comp_task_index)
        num_data = data_x.shape[0]
        num_source_class, num_target_class = self.output_sizes[task_model_index], self.output_sizes[self.current_task]

        ## Step1: Compute fake labels using pre-trained model
        raw_prob = self.eval_one_task(sess, data_x, comp_task_index, dropout_prob)

        ## Step2: Compute empirical conditional distribution of target label given source label
        joint_dist = np.zeros([num_target_class, num_source_class], dtype=np.float64)
        for c in range(num_target_class):
            class_match_index = np.squeeze(data_y)==int(c)
            joint_dist[c] = np.sum(class_match_index.reshape([num_data, 1])*raw_prob/float(num_data), axis=0)
        marginal_dist = np.mean(raw_prob, axis=0).reshape([1, num_source_class])
        cond_dist = joint_dist/marginal_dist

        ## Step3: Compute LEEP
        LEEP_score = 0.0
        for i in range(num_data):
            target_y = int(data_y[i])
            EEP = np.inner(cond_dist[target_y], raw_prob[i])
            LEEP_score += np.log(EEP)/float(num_data)
        return LEEP_score


class Lifelong_Model_EM_Algo_Frame(Lifelong_Model_Frame):
    def __init__(self, model_hyperpara, train_hyperpara):
        super().__init__(model_hyperpara, train_hyperpara)
        self.conv_sharing = []

        def _possible_choices(input_subsets):
            list_subsets = []
            for c in [True, False]:
                for elem in input_subsets:
                    list_subsets.append(elem+[c])
            return list_subsets

        self._possible_configs = [[]]
        for layer_cnt in range(self.num_conv_layers):
            self._possible_configs = _possible_choices(self._possible_configs)

        def _check_block_validity(transfer_config, block_size):
            num_blocks = len(block_size)
            indices = [sum(block_size[0:n]) for n in range(num_blocks)] + [sum(block_size)]
            cond = []
            for block_cnt in range(num_blocks):
                if block_size[block_cnt] == 1:
                    cond.append(True)
                else:
                    block_config = transfer_config[indices[block_cnt]:indices[block_cnt+1]]
                    cond.append(all(block_config) or not (any(block_config)))
            return all(cond)

        if 'layer_group_config' in model_hyperpara.keys():
            ## remove unnecessary transfer configurations if block-wise selection
            assert (sum(model_hyperpara['layer_group_config']) == self.num_conv_layers), "Given size of blocks don't match the number of conv layers!"
            for config_cnt in range(len(self._possible_configs)-1, -1, -1):
                cond = _check_block_validity(self._possible_configs[config_cnt], model_hyperpara['layer_group_config'])
                if not cond:
                    _ = self._possible_configs.pop(config_cnt)

        self.num_possible_configs = len(self._possible_configs)

    @abstractmethod
    def _shared_param_init(self):
        raise NotImplementedError

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False):
        self._shared_param_init()
        super().add_new_task(output_dim, curr_task_index, single_input_placeholder=single_input_placeholder)

    def define_eval(self):
        with tf.name_scope('Model_Eval'):
            mask = tf.reshape(tf.cast(tf.range(self.batch_size)<self.num_data_in_batch, dtype=tf.float32), [self.batch_size, 1])
            self.eval = [tf.nn.softmax(task_model[-1])*mask for task_model in self.task_models]
            self.pred = [tf.argmax(task_model[-1]*mask, 1) for task_model in self.task_models]
            if self.task_is_new:
                self.eval_for_train = [tf.nn.softmax(task_model)*mask for task_model in self.task_models[self.current_task]]
                self.pred_for_train = [tf.argmax(task_model*mask, 1) for task_model in self.task_models[self.current_task]]

                #self.likelihood = tf.stack([tf.reduce_prod(tf.boolean_mask(e, tf.one_hot(tf.cast(self.y_batch[self.current_task], tf.int32), self.output_sizes[self.current_task], on_value=True, off_value=False, dtype=tf.bool))) for e in self.eval_for_train])
                self.likelihood = tf.stack([tf.reduce_mean(tf.boolean_mask(e, tf.one_hot(tf.cast(self.y_batch[self.current_task], tf.int32), self.output_sizes[self.current_task], on_value=True, off_value=False, dtype=tf.bool))) for e in self.eval_for_train])
                self.prior = tf.Variable(np.ones(len(self._possible_configs), dtype=np.float32)/float(len(self._possible_configs)), dtype=tf.float32, trainable=False)
                posterior_tmp = tf.multiply(self.prior, self.likelihood)
                self.posterior = tf.divide(posterior_tmp, tf.reduce_sum(posterior_tmp)+1e-30)

    def define_loss(self):
        with tf.name_scope('Model_Loss'):
            self.loss = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(y_batch, tf.int32), logits=task_model[-1])) for y_batch, task_model in zip(self.y_batch, self.task_models)]
            if self.task_is_new:
                self.loss_for_train = [tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.cast(self.y_batch[self.current_task], tf.int32), logits=task_model)) for task_model in self.task_models[self.current_task]]

    def define_accuracy(self):
        with tf.name_scope('Model_Accuracy'):
            mask = tf.cast(tf.range(self.batch_size)<self.num_data_in_batch, dtype=tf.float32)
            self.accuracy = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(task_model[-1], 1), tf.cast(y_batch, tf.int64)), tf.float32)*mask) for y_batch, task_model in zip(self.y_batch, self.task_models)]
            if self.task_is_new:
                self.accuracy_for_train = [tf.reduce_sum(tf.cast(tf.equal(tf.argmax(task_model, 1), tf.cast(self.y_batch[self.current_task], tf.int64)), tf.float32)*mask) for task_model in self.task_models[self.current_task]]

    def _choose_params_for_sharing_config(self, KB_params, TS_params, conv_params, sharing_configs, num_TS_params_per_layer):
        KB_params_to_return, TS_params_to_return, conv_params_to_return = [], [], []
        for layer_cnt, (c) in enumerate(sharing_configs):
            if c:
                ## sharing
                if KB_params is not None and KB_params[layer_cnt] is not None:
                    KB_params_to_return.append(KB_params[layer_cnt])
                for tmp_cnt in range(num_TS_params_per_layer):
                    if TS_params is not None and TS_params[num_TS_params_per_layer*layer_cnt+tmp_cnt] is not None:
                        TS_params_to_return.append(TS_params[num_TS_params_per_layer*layer_cnt+tmp_cnt])
            elif conv_params is not None:
                ## task-specific
                for tmp_cnt in range(2):
                    if conv_params[2*layer_cnt+tmp_cnt] is not None:
                        conv_params_to_return.append(conv_params[2*layer_cnt+tmp_cnt])
        return KB_params_to_return, TS_params_to_return, conv_params_to_return

    def _choose_params_for_sharing_config_fillNone(self, KB_params, TS_params, conv_params, sharing_configs, num_TS_params_per_layer):
        KB_params_to_return, TS_params_to_return, conv_params_to_return = [], [], []
        for layer_cnt, (c) in enumerate(sharing_configs):
            if c:
                ## sharing
                KB_params_to_return.append(KB_params[layer_cnt])
                TS_params_to_return += [None for _ in range(num_TS_params_per_layer)] if TS_params is None else TS_params[num_TS_params_per_layer*layer_cnt:num_TS_params_per_layer*(layer_cnt+1)]
                conv_params_to_return += [None, None]
            else:
                ## task-specific
                KB_params_to_return.append(None)
                TS_params_to_return += [None for _ in range(num_TS_params_per_layer)]
                conv_params_to_return += conv_params[2*layer_cnt:2*(layer_cnt+1)]
        return KB_params_to_return, TS_params_to_return, conv_params_to_return

    def best_config(self, sess):
        prior_val = sess.run(self.prior)
        return np.argmax(prior_val)

    def _weighted_sum_two_grads(self, g1, g2, w1, w2):
        if g1 is not None and g2 is not None:
            return w1*g1+w2*g2
        elif g1 is not None:
            return w1*g1
        elif g2 is not None:
            return w2*g2
        else:
            return None

    def _weighted_sum_grads(self, grad_list, weights):
        for i in range(1, len(grad_list)):
            if i < 2:
                result = self._weighted_sum_two_grads(grad_list[i-1], grad_list[i], weights[i-1], weights[i])
            elif i < len(grad_list):
                result = self._weighted_sum_two_grads(result, grad_list[i], 1.0, weights[i])
        return result

    def eval_one_task(self, sess, data_x, task_index, dropout_prob=1.0):
        task_model_index = self.find_task_model(task_index)
        num_data, num_classes = data_x.shape[0], self.output_sizes[task_model_index]
        eval_output = np.zeros([num_data, num_classes], dtype=np.float32)

        num_batch = num_data//self.batch_size
        num_remains = num_data - self.batch_size*num_batch

        if self.task_is_new and (self.current_task == task_model_index):
            best_config = self.best_config(sess)
            eval_func = self.eval_for_train[best_config]
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
            pred_func = self.pred_for_train[best_config]
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
            acc_func = self.accuracy_for_train[best_config]
        else:
            acc_func = self.accuracy[task_model_index]

        for batch_cnt in range(num_batch):
            accuracy += sess.run(acc_func, feed_dict={self.model_input[task_model_index]: data_x[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.true_output[task_model_index]: data_y[batch_cnt*self.batch_size:(batch_cnt+1)*self.batch_size], self.dropout_prob: dropout_prob, self.num_data_in_batch: self.batch_size})
        if num_remains > 0:
            tmp_x, tmp_y = data_x_and_y_add_dummy(data_x[-num_remains:], data_y[-num_remains:], self.batch_size)
            accuracy += sess.run(acc_func, feed_dict={self.model_input[task_model_index]: tmp_x, self.true_output[task_model_index]: tmp_y, self.dropout_prob: dropout_prob, self.num_data_in_batch: num_remains})
        return float(accuracy)/float(num_data)


class Lifelong_Model_BruteForceSearch_Frame(Lifelong_Model_Frame):
    def __init__(self, model_hyperpara, train_hyperpara, model_architecture):
        super().__init__(model_hyperpara, train_hyperpara)
        self.model_architecture = model_architecture
        self.improvement_threshold = train_hyperpara['improvement_threshold']
        self.conv_sharing = []
        self.recent_task_index = -1

        def _possible_choices(input_subsets):
            list_subsets = []
            for c in [True, False]:
                for elem in input_subsets:
                    list_subsets.append(elem+[c])
            return list_subsets

        self._possible_configs = [[]]
        for layer_cnt in range(self.num_conv_layers):
            self._possible_configs = _possible_choices(self._possible_configs)
        self.num_possible_configs = len(self._possible_configs)

        self.tfsess_config = None
        self.search_on_first_task = None

    def update_param_dir(self, path_to_dir):
        self.save_param_dir = path_to_dir

    def update_tfsession_config(self, config_setup):
        self.tfsess_config = config_setup

    def _numpy_obj_to_list(self, np_obj):
        return_list = []
        for cnt in range(len(np_obj)):
            return_list.append(np_obj[cnt])
        return return_list

    @abstractmethod
    def _build_whole_model(self, params=None):
        raise NotImplementedError

    def save_params(self, file_name, sess):
        params_val = self.get_param(sess)
        savemat(file_name, {'parameter': params_val})

    def load_params(self, file_name):
        params_val = loadmat(file_name)['parameter']
        processed_params_val = self.postprocess_loaded_params(params_val)
        return processed_params_val

    @abstractmethod
    def postprocess_loaded_params(self, loaded_params):
        raise NotImplementedError

    def load_none_params(self):
        return None

    def save_training_summary(self, file_name, summary):
        savemat(file_name, {'training_summary':summary})

    def load_training_summary(self, file_name):
        loaded_summary = loadmat(file_name)['training_summary']
        processed_summary = {}
        processed_summary['history_train_error'] = loaded_summary['history_train_error'][0][0]
        processed_summary['history_validation_error'] = loaded_summary['history_validation_error'][0][0]
        processed_summary['history_test_error'] = loaded_summary['history_test_error'][0][0]
        processed_summary['history_best_test_error'] = np.squeeze(loaded_summary['history_best_test_error'][0][0])
        return processed_summary

    def define_opt(self):
        with tf.name_scope('Optimization'):
            self.grads = tf.gradients(self.loss[self.current_task], self.trainable_params)
            trainer = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate/(1.0+self.epoch*self.learn_rate_decay))
            self.update = trainer.apply_gradients(list(zip(self.grads, self.trainable_params)))

    def add_new_task(self, output_dim, curr_task_index, single_input_placeholder=False, params=None):
        self.num_trainable_var = 0

        #### placeholder of model
        self.epoch = tf.placeholder(dtype=tf.float32)
        self.dropout_prob = tf.placeholder(dtype=tf.float32)
        if single_input_placeholder:
            self.model_input = tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]])
        else:
            self.model_input = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size, self.input_size[0]*self.input_size[1]*self.input_size[2]]) for _ in range(self.num_tasks)]
        self.true_output = [tf.placeholder(dtype=tf.float32, shape=[self.batch_size]) for _ in range(self.num_tasks)]

        with tf.name_scope('Minibatch_Data'):
            if single_input_placeholder:
                self.x_batch = [tf.reshape(self.model_input, [-1]+list(self.input_size)) for _ in range(self.num_tasks)]
            else:
                self.x_batch = [tf.reshape(x, [-1]+list(self.input_size)) for x in self.model_input]
            self.y_batch = [y for y in self.true_output]

        self.task_models, self.conv_params, self.fc_params, self.params = [], [], [], []
        self._build_whole_model(params=params)
        self.define_eval()
        self.define_loss()
        self.define_accuracy()
        self.define_opt()

    def train_a_task(self, output_dim, curr_task_index, num_epoch, train_data, valid_data, test_data, save_graph=False, epoch_bias=0):
        ### single_input_placeholder=True
        if self.is_new_task(curr_task_index):
            self.num_tasks += 1
            self.output_sizes.append(output_dim)
            self.task_indices.append(curr_task_index)
            self.task_is_new = True
        else:
            self.task_is_new = False
        self.current_task = self.find_task_model(curr_task_index)

        ## Load saved parameters and process it for model initialization
        if self.recent_task_index > -1:
            param_vals_to_init = self.load_params(self.save_param_dir+'/model_parameter_task%d.mat'%(self.recent_task_index))
        else:
            param_vals_to_init = self.load_none_params()

        if self.search_on_first_task is None:
            raise ValueError

        if self.task_is_new and (self.num_tasks > 1 or self.search_on_first_task):
            validation_configwise_accuracy = np.zeros((self.num_possible_configs), dtype=np.float64)
            search_param_dir = self.save_param_dir+'/config_search_t%d'%(curr_task_index)
            mkdir(search_param_dir)
            for conf_index, (conf) in enumerate(self._possible_configs):
                ## Initialize model and accuracy_holder
                if len(self.conv_sharing) <= self.current_task:
                    self.conv_sharing.append(conf)
                else:
                    self.conv_sharing[self.current_task] = conf
                tf.reset_default_graph()

                self.add_new_task(output_dim, curr_task_index, True, params=deepcopy(param_vals_to_init))

                with tf.Session(config=self.tfsess_config) as sess:
                    sess.run(tf.global_variables_initializer())
                    print("\tConfiguration Search - %d"%(conf_index))
                    if _debug_mode:
                        self.save_params(search_param_dir+'/parameter_config%d_init.mat'%(conf_index), sess)
                    train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist, best_valid_error = self._simple_training(sess, num_epoch, train_data, valid_data, test_data, save_graph, epoch_bias, conf_index)
                    validation_configwise_accuracy[conf_index] = abs(best_valid_error)

                    ## save intermediate result (parameter and training_summary) of each tested configuration
                    self.save_params(search_param_dir+'/parameter_config%d.mat'%(conf_index), sess)
                    search_summary_temp = {}
                    search_summary_temp['history_train_error'] = train_error_hist
                    search_summary_temp['history_validation_error'] = valid_error_hist
                    search_summary_temp['history_test_error'] = test_error_hist
                    search_summary_temp['history_best_test_error'] = best_test_error_hist
                    self.save_training_summary(search_param_dir+'/summary_config%d.mat'%(conf_index), search_summary_temp)

            ## Save trained parameters in mat file (copy the parameter file with best validation performance)
            train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = self.postprocess_config_search(validation_configwise_accuracy, search_param_dir)

        else:
            tf.reset_default_graph()

            self.add_new_task(output_dim, curr_task_index, True, params=deepcopy(param_vals_to_init))

            with tf.Session(config=self.tfsess_config) as sess:
                sess.run(tf.global_variables_initializer())
                train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist, best_valid_error = self._simple_training(sess, num_epoch, train_data, valid_data, test_data, save_graph, epoch_bias)

                ## Save trained parameters in mat file
                self.save_params(self.save_param_dir+'/model_parameter_task%d.mat'%(curr_task_index), sess)

        self.recent_task_index = curr_task_index
        return train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist

    def postprocess_config_search(self, validation_configwise_accuracy, search_param_dir):
        best_conf = np.argmax(validation_configwise_accuracy)
        self.conv_sharing[self.current_task] = self._possible_configs[best_conf]
        shutil.copy2(search_param_dir+'/parameter_config%d.mat'%(best_conf), self.save_param_dir+'/model_parameter_task%d.mat'%(self.task_indices[-1]))
        loaded_best_summary = self.load_training_summary(search_param_dir+'/summary_config%d.mat'%(best_conf))
        return loaded_best_summary['history_train_error'], loaded_best_summary['history_validation_error'], loaded_best_summary['history_test_error'], loaded_best_summary['history_best_test_error']

    def _simple_training(self, sess, num_epoch, train_data, valid_data, test_data, save_graph=False, epoch_bias=0, conf_index=-1):
        task_index_in_dataset = self.task_indices[self.current_task]
        tmp = train_data[task_index_in_dataset][0]
        num_train = len(tmp) if type(tmp) is list else tmp.shape[0]
        del tmp
        indices = list(range(num_train))

        best_valid_error, test_error_at_best_epoch = np.inf, np.inf
        train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist = [], [], [], []

        num_learned_tasks = self.number_of_learned_tasks()
        num_total_tasks = len(train_data)

        if save_graph:
            if conf_index < 0:
                tfboard_writer = tf.summary.FileWriter('./graphs/%s/task%d'%(self.model_architecture, num_learned_tasks), sess.graph)
            else:
                tfboard_writer = tf.summary.FileWriter('./graphs/%s/task%d/config%d'%(self.model_architecture, num_learned_tasks, conf_index), sess.graph)
            tfboard_writer.close()

        model_error = self.accuracy
        batch_size = self.batch_size
        for learning_step in range(0 if (num_learned_tasks < 2 and self.task_is_new) else 1, num_epoch+1):
            if learning_step > 0:
                shuffle(indices)

                for batch_cnt in range(num_train//batch_size):
                    batch_train_x = train_data[task_index_in_dataset][0][indices[batch_cnt*batch_size:(batch_cnt+1)*batch_size], :]
                    batch_train_y = train_data[task_index_in_dataset][1][indices[batch_cnt*batch_size:(batch_cnt+1)*batch_size]]

                    sess.run(self.update, feed_dict={self.model_input: batch_train_x, self.true_output[self.current_task]: batch_train_y, self.epoch: learning_step+epoch_bias-1, self.dropout_prob: 0.5})

            train_error_tmp = [0.0 for _ in range(num_total_tasks)]
            validation_error_tmp = [0.0 for _ in range(num_total_tasks)]
            test_error_tmp = [0.0 for _ in range(num_total_tasks)]

            for task_index_to_eval in self.task_indices:
                task_model_index_to_eval = self.find_task_model(task_index_to_eval)
                num_train_to_eval = len(train_data[task_index_to_eval][0]) if type(train_data[task_index_to_eval][0]) is list else train_data[task_index_to_eval][0].shape[0]
                num_valid_to_eval = len(valid_data[task_index_to_eval][0]) if type(valid_data[task_index_to_eval][0]) is list else valid_data[task_index_to_eval][0].shape[0]
                num_test_to_eval = len(test_data[task_index_to_eval][0]) if type(test_data[task_index_to_eval][0]) is list else test_data[task_index_to_eval][0].shape[0]


                for batch_cnt in range(num_train_to_eval//batch_size):
                    train_error_tmp[task_index_to_eval] = train_error_tmp[task_index_to_eval] + sess.run(model_error[task_model_index_to_eval], feed_dict={self.model_input: train_data[task_index_to_eval][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], self.true_output[task_model_index_to_eval]: train_data[task_index_to_eval][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size], self.dropout_prob: 1.0})
                train_error_tmp[task_index_to_eval] = train_error_tmp[task_index_to_eval]/((num_train_to_eval//batch_size)*batch_size)

                for batch_cnt in range(num_valid_to_eval//batch_size):
                    validation_error_tmp[task_index_to_eval] = validation_error_tmp[task_index_to_eval] + sess.run(model_error[task_model_index_to_eval], feed_dict={self.model_input: valid_data[task_index_to_eval][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], self.true_output[task_model_index_to_eval]: valid_data[task_index_to_eval][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size], self.dropout_prob: 1.0})
                validation_error_tmp[task_index_to_eval] = validation_error_tmp[task_index_to_eval]/((num_valid_to_eval//batch_size)*batch_size)

                for batch_cnt in range(num_test_to_eval//batch_size):
                    test_error_tmp[task_index_to_eval] = test_error_tmp[task_index_to_eval] + sess.run(model_error[task_model_index_to_eval], feed_dict={self.model_input: test_data[task_index_to_eval][0][batch_cnt*batch_size:(batch_cnt+1)*batch_size, :], self.true_output[task_model_index_to_eval]: test_data[task_index_to_eval][1][batch_cnt*batch_size:(batch_cnt+1)*batch_size], self.dropout_prob: 1.0})
                test_error_tmp[task_index_to_eval] = test_error_tmp[task_index_to_eval]/((num_test_to_eval//batch_size)*batch_size)

            ## for classification, error_tmp is actually ACCURACY, thus, change the sign for checking improvement
            train_error, valid_error, test_error = -(sum(train_error_tmp)/(num_learned_tasks)), -(sum(validation_error_tmp)/(num_learned_tasks)), -(sum(test_error_tmp)/(num_learned_tasks))
            train_error_to_compare, valid_error_to_compare, test_error_to_compare = -train_error_tmp[task_index_in_dataset], -validation_error_tmp[task_index_in_dataset], -test_error_tmp[task_index_in_dataset]

            #### error related process
            print('epoch %d - Train : %f, Validation : %f' % (learning_step+epoch_bias, abs(train_error_to_compare), abs(valid_error_to_compare)))

            if valid_error_to_compare < best_valid_error:
                str_temp = ''
                if valid_error_to_compare < best_valid_error * self.improvement_threshold:
                    str_temp = '\t<<'
                best_valid_error, best_epoch = valid_error_to_compare, learning_step
                test_error_at_best_epoch = test_error_to_compare
                print('\t\t\t\t\t\t\tTest : %f%s' % (abs(test_error_at_best_epoch), str_temp))

            train_error_hist.append(train_error_tmp + [abs(train_error)])
            valid_error_hist.append(validation_error_tmp + [abs(valid_error)])
            test_error_hist.append(test_error_tmp + [abs(test_error)])
            best_test_error_hist.append(abs(test_error_at_best_epoch))
        return (train_error_hist, valid_error_hist, test_error_hist, best_test_error_hist, best_valid_error)