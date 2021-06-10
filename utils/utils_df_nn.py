import numpy as np
import tensorflow as tf

from utils.utils import *
from utils.utils_nn import *


###########################################################
#####         functions to generate parameter         #####
###########################################################

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
def new_ELLA_cnn_deconv_TS_param(shape, layer_number, task_number, reg_type):
    ts_w_name, ts_b_name, ts_p_name = 'TS_DeconvW0_'+str(layer_number)+'_'+str(task_number), 'TS_Deconvb0_'+str(layer_number)+'_'+str(task_number), 'TS_Convb0_'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=ts_w_name, shape=shape[0], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_b_name, shape=shape[1], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_p_name, shape=shape[2], dtype=tf.float32, regularizer=reg_type)]

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

#### function to generate task-specific parameters for ELLA_tensorfactor layer
def new_ELLA_cnn_deconv_tensordot_TS_param2(shape, layer_number, task_number, reg_type):
    ts_w_name, ts_b_name, ts_k_name, ts_k_name2, ts_p_name = 'TS_DeconvW0_'+str(layer_number)+'_'+str(task_number), 'TS_Deconvb0_'+str(layer_number)+'_'+str(task_number), 'TS_tdot_W1_'+str(layer_number)+'_'+str(task_number), 'TS_tdot_W2_'+str(layer_number)+'_'+str(task_number), 'TS_tdot_b0_'+str(layer_number)+'_'+str(task_number)
    return [tf.get_variable(name=ts_w_name, shape=shape[0], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_b_name, shape=shape[1], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_k_name, shape=shape[2], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_k_name2, shape=shape[3], dtype=tf.float32, regularizer=reg_type), tf.get_variable(name=ts_p_name, shape=shape[4], dtype=tf.float32, regularizer=reg_type)]


###############################################################
##### functions for adding ELLA network (CNN/Deconv ver)  #####
###############################################################

#### function to generate convolutional layer with shared knowledge base
#### KB_size : [filter_height(and width), num_of_channel]
#### TS_size : deconv_filter_height(and width)
#### TS_stride_size : [stride_in_height, stride_in_width]
def new_ELLA_cnn_deconv_layer(layer_input, k_size, ch_size, stride_size, KB_size, TS_size, TS_stride_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_size=None, skip_connect_input=None):
    assert (k_size[0] == k_size[1] and k_size[0] == (KB_size[0]-1)*TS_stride_size[0]+1), "CNN kernel size does not match the output size of Deconv from KB"

    with tf.name_scope('ELLA_cdnn_KB'):
        if KB_param is None:
            ## KB \in R^{1 \times h \times w \times c}
            KB_param = new_ELLA_KB_param([1, KB_size[0], KB_size[0], KB_size[1]], layer_num, task_num, KB_reg_type)
        if TS_param is None:
            ## TS1 : Deconv W \in R^{h \times w \times ch_in*ch_out \times c}
            ## TS2 : Deconv bias \in R^{ch_out}
            TS_param = new_ELLA_cnn_deconv_TS_param([[TS_size, TS_size, ch_size[0]*ch_size[1], KB_size[1]], [1, 1, 1, ch_size[0]*ch_size[1]], [ch_size[1]]], layer_num, task_num, TS_reg_type)

    with tf.name_scope('ELLA_cdnn_TS'):
        para_tmp = tf.add(tf.nn.conv2d_transpose(KB_param, TS_param[0], [1, k_size[0], k_size[1], ch_size[0]*ch_size[1]], strides=[1, TS_stride_size[0], TS_stride_size[1], 1]), TS_param[1])
        if para_activation_fn is not None:
            para_tmp = para_activation_fn(para_tmp)

        W, b = tf.reshape(para_tmp, k_size+ch_size), TS_param[2]

    layer_eqn, _ = new_cnn_layer(layer_input, k_size+ch_size, stride_size=stride_size, activation_fn=activation_fn, weight=W, bias=b, padding_type=padding_type, max_pooling=max_pool, pool_size=pool_size, skip_connect_input=skip_connect_input)
    return layer_eqn, [KB_param], TS_param, [W, b]


#### function to generate network of convolutional layers with shared knowledge base
def new_ELLA_cnn_deconv_net(net_input, k_sizes, ch_sizes, stride_sizes, KB_sizes, TS_sizes, TS_stride_sizes, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_params=None, TS_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0], task_index=0, skip_connections=[]):
    _num_TS_param_per_layer = 3

    ## first element : make new KB&TS / second element : make new TS / third element : not make new para
    control_flag = [(KB_params is None and TS_params is None), (not (KB_params is None) and (TS_params is None)), not (KB_params is None or TS_params is None)]
    if control_flag[1]:
        TS_params = []
    elif control_flag[0]:
        KB_params, TS_params = [], []
    cnn_gen_params=[]

    layers_for_skip, next_skip_connect = [net_input], None
    with tf.name_scope('ELLA_cdnn_net'):
        layers = []
        for layer_cnt in range(len(k_sizes)//2):
            next_skip_connect = skip_connections.pop(0) if (len(skip_connections) > 0 and next_skip_connect is None) else next_skip_connect
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

            if layer_cnt == 0 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_ELLA_cnn_deconv_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif layer_cnt == 0 and control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp = new_ELLA_cnn_deconv_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif layer_cnt == 0 and control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp = new_ELLA_cnn_deconv_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_ELLA_cnn_deconv_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp = new_ELLA_cnn_deconv_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp = new_ELLA_cnn_deconv_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[layer_cnt], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)

            layers.append(layer_tmp)
            layers_for_skip.append(layer_tmp)
            cnn_gen_params = cnn_gen_params + cnn_gen_para_tmp
            if control_flag[1]:
                TS_params = TS_params + TS_para_tmp
            elif control_flag[0]:
                KB_params = KB_params + KB_para_tmp
                TS_params = TS_params + TS_para_tmp
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
    return (layers, KB_params, TS_params, cnn_gen_params, output_dim)


#### function to generate network of cnn->ffnn
def new_ELLA_cnn_deconv_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, task_index=0, skip_connections=[]):
    ## add CNN layers
    cnn_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, cnn_output_dim = new_ELLA_cnn_deconv_net(net_input, k_sizes, ch_sizes, stride_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_params=cnn_KB_params, TS_params=cnn_TS_params, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, task_index=task_index, skip_connections=skip_connections)

    ## add fc layers
    #fc_model, fc_params = new_fc_net(cnn_model[-1], [cnn_output_dim[0]]+fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net')
    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net')

    return (cnn_model+fc_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, fc_params)




###########################################################################
##### functions for adding ELLA network (CNN/Deconv & Tensordot ver)  #####
###########################################################################

#### KB_size : [filter_height(and width), num_of_channel]
#### TS_size : [deconv_filter_height(and width), deconv_filter_channel]
#### TS_stride_size : [stride_in_height, stride_in_width]
def new_ELLA_cnn_deconv_tensordot_layer(layer_input, k_size, ch_size, stride_size, KB_size, TS_size, TS_stride_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_size=None, skip_connect_input=None, highway_connect_type=0, highway_W=None, highway_b=None, trainable=True, trainable_KB=True):
    assert (k_size[0] == k_size[1] and k_size[0] == (KB_size[0]-1)*TS_stride_size[0]+1), "CNN kernel size does not match the output size of Deconv from KB"

    with tf.name_scope('ELLA_cdnn_KB'):
        ## KB \in R^{1 \times h \times w \times c}
        KB_param = new_ELLA_KB_param([1, KB_size[0], KB_size[0], KB_size[1]], layer_num, task_num, KB_reg_type, KB_param, trainable=trainable_KB)

        ## TS1 : Deconv W \in R^{h \times w \times kb_c_out \times c}
        ## TS2 : Deconv bias \in R^{kb_c_out}
        ## TS3 : tensor W \in R^{kb_c_out \times ch_in \times ch_out}
        ## TS4 : Conv bias \in R^{ch_out}
        TS_param = new_ELLA_cnn_deconv_tensordot_TS_param([[TS_size[0], TS_size[0], TS_size[1], KB_size[1]], [1, 1, 1, TS_size[1]], [TS_size[1], ch_size[0], ch_size[1]], [ch_size[1]]], layer_num, task_num, TS_reg_type, [None, None, None, None] if TS_param is None else TS_param, trainable=trainable)

    with tf.name_scope('DFCNN_param_gen'):
        para_tmp = tf.add(tf.nn.conv2d_transpose(KB_param, TS_param[0], [1, k_size[0], k_size[1], TS_size[1]], strides=[1, TS_stride_size[0], TS_stride_size[1], 1]), TS_param[1])
        para_tmp = tf.reshape(para_tmp, [k_size[0], k_size[1], TS_size[1]])
        if para_activation_fn is not None:
            para_tmp = para_activation_fn(para_tmp)
        W = tf.tensordot(para_tmp, TS_param[2], [[2], [0]])
        b = TS_param[3]

    ## HighwayNet's skip connection
    highway_params, gate = [], None
    if highway_connect_type > 0:
        with tf.name_scope('highway_connection'):
            if highway_connect_type == 1:
                x = layer_input
                if highway_W is None:
                    highway_W = new_weight([k_size[0], k_size[1], ch_size[0], ch_size[1]])
                if highway_b is None:
                    highway_b = new_bias([ch_size[1]], init_val=-2.0)
                gate, _ = new_cnn_layer(x, k_size+ch_size, stride_size=stride_size, activation_fn=None, weight=highway_W, bias=highway_b, padding_type=padding_type, max_pooling=False)
            elif highway_connect_type == 2:
                x = tf.reshape(layer_input, [-1, int(layer_input.shape[1]*layer_input.shape[2]*layer_input.shape[3])])
                if highway_W is None:
                    highway_W = new_weight([int(x.shape[1]), 1])
                if highway_b is None:
                    highway_b = new_bias([1], init_val=-2.0)
                gate = tf.broadcast_to(tf.stack([tf.stack([tf.matmul(x, highway_W) + highway_b], axis=2)], axis=3), layer_input.get_shape())
            gate = tf.nn.sigmoid(gate)
        highway_params = [highway_W, highway_b]

    layer_eqn, _ = new_cnn_layer(layer_input, k_size+ch_size, stride_size=stride_size, activation_fn=activation_fn, weight=W, bias=b, padding_type=padding_type, max_pooling=max_pool, pool_size=pool_size, skip_connect_input=skip_connect_input, highway_connect_type=highway_connect_type, highway_gate=gate)
    return layer_eqn, [KB_param], TS_param, [W, b], highway_params


#### function to generate network of convolutional layers with shared knowledge base
def new_ELLA_cnn_deconv_tensordot_net(net_input, k_sizes, ch_sizes, stride_sizes, KB_sizes, TS_sizes, TS_stride_sizes, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_params=None, TS_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0], task_index=0, skip_connections=[]):
    _num_TS_param_per_layer = 4

    ## first element : make new KB&TS / second element : make new TS / third element : not make new para / fourth element : make new KB
    control_flag = [(KB_params is None and TS_params is None), (not (KB_params is None) and (TS_params is None)), not (KB_params is None or TS_params is None), ((KB_params is None) and not (TS_params is None))]
    if control_flag[1]:
        TS_params = []
    elif control_flag[3]:
        KB_params = []
    elif control_flag[0]:
        KB_params, TS_params = [], []
    cnn_gen_params = []

    layers_for_skip, next_skip_connect = [net_input], None
    with tf.name_scope('ELLA_cdnn_net'):
        layers = []
        for layer_cnt in range(len(k_sizes)//2):
            next_skip_connect = skip_connections.pop(0) if (len(skip_connections) > 0 and next_skip_connect is None) else next_skip_connect
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

            if layer_cnt == 0 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, _ = new_ELLA_cnn_deconv_tensordot_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif layer_cnt == 0 and control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp, _ = new_ELLA_cnn_deconv_tensordot_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif layer_cnt == 0 and control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp, _ = new_ELLA_cnn_deconv_tensordot_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif layer_cnt == 0 and control_flag[3]:
                layer_tmp, KB_para_tmp, _, cnn_gen_para_tmp, _ = new_ELLA_cnn_deconv_tensordot_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, _ = new_ELLA_cnn_deconv_tensordot_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp, _ = new_ELLA_cnn_deconv_tensordot_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp, _ = new_ELLA_cnn_deconv_tensordot_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[3]:
                layer_tmp, KB_para_tmp, _, cnn_gen_para_tmp, _ = new_ELLA_cnn_deconv_tensordot_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)

            layers.append(layer_tmp)
            layers_for_skip.append(layer_tmp)
            cnn_gen_params = cnn_gen_params + cnn_gen_para_tmp
            if control_flag[1]:
                TS_params = TS_params + TS_para_tmp
            elif control_flag[3]:
                KB_params = KB_params + KB_para_tmp
            elif control_flag[0]:
                KB_params = KB_params + KB_para_tmp
                TS_params = TS_params + TS_para_tmp
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
    return (layers, KB_params, TS_params, cnn_gen_params, output_dim)


#### function to generate network of cnn (with shared KB through deconv)-> simple ffnn
def new_ELLA_cnn_deconv_tensordot_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, task_index=0, skip_connections=[]):
    ## add CNN layers
    cnn_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, cnn_output_dim = new_ELLA_cnn_deconv_tensordot_net(net_input, k_sizes, ch_sizes, stride_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_params=cnn_KB_params, TS_params=cnn_TS_params, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, task_index=task_index, skip_connections=skip_connections)

    ## add fc layers
    #fc_model, fc_params = new_fc_net(cnn_model[-1], [cnn_output_dim[0]]+fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net')
    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net')

    return (cnn_model+fc_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, fc_params)


###########################################################################
##### functions for adding ELLA network (CNN/Deconv & Tensordot ver2)  #####
###########################################################################
#### KB_size : [filter_height(and width), num_of_channel0, num_of_channel1]
#### TS_size : [deconv_filter_height(and width), deconv_filter_channel]
#### TS_stride_size : [stride_in_height, stride_in_width]
def new_ELLA_cnn_deconv_tensordot_layer2(layer_input, k_size, ch_size, stride_size, KB_size, TS_size, TS_stride_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_size=None, skip_connect_input=None):
    assert (k_size[0] == k_size[1] and k_size[0] == (KB_size[0]-1)*TS_stride_size[0]+1), "CNN kernel size does not match the output size of Deconv from KB"

    with tf.name_scope('ELLA_cdnn_KB'):
        if KB_param is None:
            ## KB \in R^{d \times h \times w \times c}
            KB_param = new_ELLA_KB_param([KB_size[1], KB_size[0], KB_size[0], KB_size[2]], layer_num, task_num, KB_reg_type)
        if TS_param is None:
            ## TS1 : Deconv W \in R^{h \times w \times kb_c_out \times c}
            ## TS2 : Deconv bias \in R^{kb_c_out}
            ## TS3 : tensor W \in R^{d \times ch_in}
            ## TS4 : tensor W \in R^{kb_c_out \times ch_out}
            ## TS5 : Conv bias \in R^{ch_out}
            TS_param = new_ELLA_cnn_deconv_tensordot_TS_param2([[TS_size[0], TS_size[0], TS_size[1], KB_size[2]], [1, 1, 1, TS_size[1]], [KB_size[1], ch_size[0]], [TS_size[1], ch_size[1]], [1, 1, 1, ch_size[1]]], layer_num, task_num, TS_reg_type)

    with tf.name_scope('ELLA_cdnn_TS'):
        para_tmp = tf.add(tf.nn.conv2d_transpose(KB_param, TS_param[0], [KB_size[1], k_size[0], k_size[1], TS_size[1]], strides=[1, TS_stride_size[0], TS_stride_size[1], 1]), TS_param[1])
        if para_activation_fn is not None:
            para_tmp = para_activation_fn(para_tmp)
        para_tmp = tf.tensordot(para_tmp, TS_param[2], [[0], [0]])
        W = tf.tensordot(para_tmp, TS_param[3], [[2], [0]])
        b = TS_param[4]

    layer_eqn, _ = new_cnn_layer(layer_input, k_size+ch_size, stride_size=stride_size, activation_fn=activation_fn, weight=W, bias=b, padding_type=padding_type, max_pooling=max_pool, pool_size=pool_size, skip_connect_input=skip_connect_input)
    return layer_eqn, [KB_param], TS_param, [W, b]


#### function to generate network of convolutional layers with shared knowledge base
def new_ELLA_cnn_deconv_tensordot_net2(net_input, k_sizes, ch_sizes, stride_sizes, KB_sizes, TS_sizes, TS_stride_sizes, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_params=None, TS_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0], task_index=0, skip_connections=[]):
    _num_TS_param_per_layer = 5

    ## first element : make new KB&TS / second element : make new TS / third element : not make new para / fourth element : make new KB
    control_flag = [(KB_params is None and TS_params is None), (not (KB_params is None) and (TS_params is None)), not (KB_params is None or TS_params is None), ((KB_params is None) and not (TS_params is None))]
    if control_flag[1]:
        TS_params = []
    elif control_flag[3]:
        KB_params = []
    elif control_flag[0]:
        KB_params, TS_params = [], []
    cnn_gen_params = []

    layers_for_skip, next_skip_connect = [net_input], None
    with tf.name_scope('ELLA_cdnn_net'):
        layers = []
        for layer_cnt in range(len(k_sizes)//2):
            next_skip_connect = skip_connections.pop(0) if (len(skip_connections) > 0 and next_skip_connect is None) else next_skip_connect
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

            if layer_cnt == 0 and control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif layer_cnt == 0 and control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif layer_cnt == 0 and control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif layer_cnt == 0 and control_flag[3]:
                layer_tmp, KB_para_tmp, _, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[0]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[1]:
                layer_tmp, _, TS_para_tmp, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[2]:
                layer_tmp, _, _, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=KB_params[layer_cnt], TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)
            elif control_flag[3]:
                layer_tmp, KB_para_tmp, _, cnn_gen_para_tmp = new_ELLA_cnn_deconv_tensordot_layer2(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], KB_sizes[3*layer_cnt:3*(layer_cnt+1)], TS_sizes[2*layer_cnt:2*(layer_cnt+1)], TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=para_activation_fn, KB_param=None, TS_param=TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input)

            layers.append(layer_tmp)
            layers_for_skip.append(layer_tmp)
            cnn_gen_params = cnn_gen_params + cnn_gen_para_tmp
            if control_flag[1]:
                TS_params = TS_params + TS_para_tmp
            elif control_flag[3]:
                KB_params = KB_params + KB_para_tmp
            elif control_flag[0]:
                KB_params = KB_params + KB_para_tmp
                TS_params = TS_params + TS_para_tmp
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
    return (layers, KB_params, TS_params, cnn_gen_params, output_dim)


#### function to generate network of cnn (with shared KB through deconv)-> simple ffnn
def new_ELLA_cnn_deconv_tensordot_fc_net2(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, task_index=0, skip_connections=[]):
    ## add CNN layers
    cnn_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, cnn_output_dim = new_ELLA_cnn_deconv_tensordot_net2(net_input, k_sizes, ch_sizes, stride_sizes, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_params=cnn_KB_params, TS_params=cnn_TS_params, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, task_index=task_index, skip_connections=skip_connections)

    ## add fc layers
    #fc_model, fc_params = new_fc_net(cnn_model[-1], [cnn_output_dim[0]]+fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net')
    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net')

    return (cnn_model+fc_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, fc_params)


##############################################################################################################
####  functions for Conv-FC nets whose conv layers are freely set to shared across tasks by DeconvFactor  ####
##############################################################################################################
def new_ELLA_flexible_cnn_deconv_tensordot_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_sharing, cnn_KB_sizes, cnn_TS_sizes, cnn_TS_stride_sizes, cnn_activation_fn=tf.nn.relu, cnn_para_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, task_index=0, skip_connections=[], highway_connect_type=0, cnn_highway_params=None, trainable=True, trainable_KB=True):
    _num_TS_param_per_layer = 4

    num_conv_layers = [len(k_sizes)//2, len(ch_sizes)-1, len(stride_sizes)//2, len(cnn_sharing), len(cnn_KB_sizes)//2, len(cnn_TS_sizes)//2, len(cnn_TS_stride_sizes)//2]
    assert (all([(num_conv_layers[i]==num_conv_layers[i+1]) for i in range(len(num_conv_layers)-1)])), "Parameters related to conv layers are wrong!"
    num_conv_layers = num_conv_layers[0]
    '''
    if cnn_KB_params is not None:
        assert (len(cnn_KB_params) == 1), "Given init value of KB (last layer) is wrong!"
    if cnn_TS_params is not None:
        assert (len(cnn_TS_params) == 4), "Given init value of TS (last layer) is wrong!"
    '''

    ## add CNN layers
    ## first element : make new KB&TS / second element : make new TS / third element : not make new para / fourth element : make new KB
    control_flag = [(cnn_KB_params is None and cnn_TS_params is None), (not (cnn_KB_params is None) and (cnn_TS_params is None)), not (cnn_KB_params is None or cnn_TS_params is None), ((cnn_KB_params is None) and not (cnn_TS_params is None))]
    if control_flag[1]:
        cnn_TS_params = []
    elif control_flag[3]:
        cnn_KB_params = []
    elif control_flag[0]:
        cnn_KB_params, cnn_TS_params = [], []
    cnn_gen_params = []

    if cnn_params is None:
        cnn_params = [None for _ in range(2*num_conv_layers)]

    layers_for_skip, next_skip_connect = [net_input], None
    with tf.name_scope('Hybrid_DFCNN'):
        cnn_model, cnn_params_to_return, cnn_highway_params_to_return = [], [], []
        cnn_KB_to_return, cnn_TS_to_return = [], []
        for layer_cnt in range(num_conv_layers):
            KB_para_tmp, TS_para_tmp, para_tmp = [None], [None for _ in range(_num_TS_param_per_layer)], [None, None]
            highway_para_tmp = [None, None] if cnn_highway_params is None else cnn_highway_params[2*layer_cnt:2*(layer_cnt+1)]
            cnn_gen_para_tmp = [None, None]

            next_skip_connect = skip_connections.pop(0) if (len(skip_connections) > 0 and next_skip_connect is None) else next_skip_connect
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
                if control_flag[0] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[1] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[2] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[3] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif (not cnn_sharing[layer_cnt]):
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable)
            else:
                if control_flag[0] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[1] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=None, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[2] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif control_flag[3] and cnn_sharing[layer_cnt]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_ELLA_cnn_deconv_tensordot_layer(cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], cnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], cnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=cnn_activation_fn, para_activation_fn=cnn_para_activation_fn, KB_param=None, TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
                elif (not cnn_sharing[layer_cnt]):
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=cnn_model[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable)

            cnn_model.append(layer_tmp)
            layers_for_skip.append(layer_tmp)
            cnn_KB_to_return = cnn_KB_to_return + KB_para_tmp
            cnn_TS_to_return = cnn_TS_to_return + TS_para_tmp
            cnn_params_to_return = cnn_params_to_return + para_tmp
            cnn_gen_params = cnn_gen_params + cnn_gen_para_tmp
            cnn_highway_params_to_return = cnn_highway_params_to_return + highway_para_tmp
            if layer_cnt == skip_connect_out:
                next_skip_connect = None

        #### flattening output
        output_dim = [int(cnn_model[-1].shape[1]*cnn_model[-1].shape[2]*cnn_model[-1].shape[3])]
        cnn_model.append(tf.reshape(cnn_model[-1], [-1, output_dim[0]]))

        #### add dropout layer
        if dropout:
            cnn_model.append(tf.nn.dropout(cnn_model[-1], dropout_prob))

    ## add fc layers
    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net', trainable=trainable)

    #return (cnn_model+fc_model, cnn_KB_params, cnn_TS_params, cnn_gen_params, cnn_params_to_return, cnn_highway_params_to_return, fc_params)
    return (cnn_model+fc_model, cnn_KB_to_return, cnn_TS_to_return, cnn_gen_params, cnn_params_to_return, cnn_highway_params_to_return, fc_params)


#### function to generate DARTS-based network for selective sharing on DF-CNN
def new_darts_dfcnn_layer(layer_input, k_size, ch_size, stride_size, KB_size, TS_size, TS_stride_size, layer_num, task_num, activation_fn=tf.nn.relu, para_activation_fn=tf.nn.relu, KB_param=None, TS_param=None, conv_param=None, select_param=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pooling=False, pool_size=None, trainable=True, skip_connect_input=None, name_scope='darts_dfcnn_layer', use_numpy_var_in_graph=False):
    with tf.name_scope(name_scope):
        ## init DF-CNN KB params
        if KB_param is None or (type(KB_param) == np.ndarray and not use_numpy_var_in_graph):
            KB_param = new_ELLA_KB_param([1, KB_size[0], KB_size[0], KB_size[1]], layer_num, task_num, KB_reg_type, KB_param, trainable=trainable)

        ## init DF-CNN task-specific mapping params
        if TS_param is None or (type(TS_param) == np.ndarray and not use_numpy_var_in_graph):
            TS_param = new_ELLA_cnn_deconv_tensordot_TS_param([[TS_size[0], TS_size[0], TS_size[1], KB_size[1]], [1, 1, 1, TS_size[1]], [TS_size[1], ch_size[0], ch_size[1]], [ch_size[1]]], layer_num, task_num, TS_reg_type, [None, None, None, None] if TS_param is None else TS_param, trainable=trainable)

        ## init task-specific conv params
        if conv_param is None:
            conv_param = [new_weight(shape=k_size+ch_size, trainable=trainable), new_bias(shape=[ch_size[-1]], trainable=trainable)]
        else:
            if conv_param[0] is None or (type(conv_param[0]) == np.ndarray and not use_numpy_var_in_graph):
                conv_param[0] = new_weight(shape=k_size+ch_size, init_tensor=conv_param[0], trainable=trainable)
            if conv_param[1] is None or (type(conv_param[1]) == np.ndarray and not use_numpy_var_in_graph):
                conv_param[1] = new_bias(shape=[ch_size[-1]], init_tensor=conv_param[1], trainable=trainable)

        ## init DARTS-selection params
        if select_param is None:
            select_param = new_weight(shape=[2], init_tensor=np.zeros(2, dtype=np.float32), trainable=trainable)
        elif (type(select_param) == np.ndarray) and not use_numpy_var_in_graph:
            select_param = new_weight(shape=[2], init_tensor=select_param, trainable=trainable)

        with tf.name_scope('DFCNN_param_gen'):
            para_tmp = tf.add(tf.nn.conv2d_transpose(KB_param, TS_param[0], [1, k_size[0], k_size[1], TS_size[1]], strides=[1, TS_stride_size[0], TS_stride_size[1], 1]), TS_param[1])
            para_tmp = tf.reshape(para_tmp, [k_size[0], k_size[1], TS_size[1]])
            if para_activation_fn is not None:
                para_tmp = para_activation_fn(para_tmp)
            W = tf.tensordot(para_tmp, TS_param[2], [[2], [0]])
            b = TS_param[3]

        mixing_weight = tf.reshape(tf.nn.softmax(select_param), [2,1])
        shared_conv_layer = tf.nn.conv2d(layer_input, W, strides=stride_size, padding=padding_type) + b
        TS_conv_layer = tf.nn.conv2d(layer_input, conv_param[0], strides=stride_size, padding=padding_type) + conv_param[1]

        if skip_connect_input is not None:
            shape1, shape2 = shared_conv_layer.get_shape().as_list(), skip_connect_input.get_shape().as_list()
            assert (len(shape1) == len(shape2)), "Shape of layer's output and input of skip connection do not match!"
            assert (all([(x==y) for (x, y) in zip(shape1, shape2)])), "Shape of layer's output and input of skip connection do NOT match!"
            shared_conv_layer = shared_conv_layer + skip_connect_input
            TS_conv_layer = TS_conv_layer + skip_connect_input

        if not (activation_fn is None):
            shared_conv_layer = activation_fn(shared_conv_layer)
            TS_conv_layer = activation_fn(TS_conv_layer)

        mixed_conv_temp = tf.tensordot(tf.stack([TS_conv_layer, shared_conv_layer], axis=4), mixing_weight, axes=[[4], [0]])
        conv_layer = tf.reshape(mixed_conv_temp, mixed_conv_temp.get_shape()[0:-1])

        if max_pooling and (pool_size[1] > 1 or pool_size[2] > 1):
            layer = tf.nn.max_pool(conv_layer, ksize=pool_size, strides=pool_size, padding=padding_type)
        else:
            layer = conv_layer
    return (layer, [KB_param], TS_param, conv_param, [select_param])

def new_darts_dfcnn_net(net_input, k_sizes, ch_sizes, stride_sizes, dfcnn_KB_sizes, dfcnn_TS_sizes, dfcnn_TS_stride_sizes, activation_fn=tf.nn.relu, dfcnn_TS_activation_fn=tf.nn.relu, dfcnn_KB_params=None, dfcnn_TS_params=None, cnn_TS_params=None, select_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, trainable=True, task_index=0, skip_connections=[], use_numpy_var_in_graph=False):
    _num_TS_param_per_layer = 4
    num_conv_layers = [len(k_sizes)//2, len(ch_sizes)-1, len(stride_sizes)//2, len(dfcnn_KB_sizes)//2, len(dfcnn_TS_sizes)//2, len(dfcnn_TS_stride_sizes)//2]
    assert (all([(num_conv_layers[i]==num_conv_layers[i+1]) for i in range(len(num_conv_layers)-1)])), "Parameters related to conv layers are wrong!"
    num_conv_layers = num_conv_layers[0]

    ## first element : make new KB&TS / second element : make new TS / third element : not make new para / fourth element : make new KB
    control_flag = [(dfcnn_KB_params is None and dfcnn_TS_params is None), (not (dfcnn_KB_params is None) and (dfcnn_TS_params is None)), not (dfcnn_KB_params is None or dfcnn_TS_params is None), ((dfcnn_KB_params is None) and not (dfcnn_TS_params is None))]

    if cnn_TS_params is None:
        cnn_TS_params = [None for _ in range(2*num_conv_layers)]
    else:
        assert(len(cnn_TS_params) == 2*num_conv_layers), "Check given parameters!"
    if select_params is None:
        select_params = [None for _ in range(num_conv_layers)]

    layers_for_skip, next_skip_connect = [net_input], None
    layers, dfcnn_shared_params_return, dfcnn_TS_params_return, cnn_TS_params_return, select_params_return = [], [], [], [], []
    with tf.name_scope('DARTS_DFCNN_net'):
        for layer_cnt in range(num_conv_layers):
            next_skip_connect = skip_connections.pop(0) if (len(skip_connections) > 0 and next_skip_connect is None) else None
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
                if control_flag[0]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_TS_para_tmp, select_para_tmp = new_darts_dfcnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], dfcnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=dfcnn_TS_activation_fn, KB_param=None, TS_param=None, conv_param=cnn_TS_params[2*layer_cnt:2*(layer_cnt+1)], select_param=select_params[layer_cnt], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, skip_connect_input=processed_skip_connect_input, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif control_flag[1]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_TS_para_tmp, select_para_tmp = new_darts_dfcnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], dfcnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=dfcnn_TS_activation_fn, KB_param=dfcnn_KB_params[layer_cnt], TS_param=None, conv_param=cnn_TS_params[2*layer_cnt:2*(layer_cnt+1)], select_param=select_params[layer_cnt], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, skip_connect_input=processed_skip_connect_input, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif control_flag[2]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_TS_para_tmp, select_para_tmp = new_darts_dfcnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], dfcnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=dfcnn_TS_activation_fn, KB_param=dfcnn_KB_params[layer_cnt], TS_param=dfcnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], conv_param=cnn_TS_params[2*layer_cnt:2*(layer_cnt+1)], select_param=select_params[layer_cnt], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, skip_connect_input=processed_skip_connect_input, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif control_flag[3]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_TS_para_tmp, select_para_tmp = new_darts_dfcnn_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], dfcnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=dfcnn_TS_activation_fn, KB_param=None, TS_param=dfcnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], conv_param=cnn_TS_params[2*layer_cnt:2*(layer_cnt+1)], select_param=select_params[layer_cnt], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, skip_connect_input=processed_skip_connect_input, use_numpy_var_in_graph=use_numpy_var_in_graph)
            else:
                if control_flag[0]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_TS_para_tmp, select_para_tmp = new_darts_dfcnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], dfcnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=dfcnn_TS_activation_fn, KB_param=None, TS_param=None, conv_param=cnn_TS_params[2*layer_cnt:2*(layer_cnt+1)], select_param=select_params[layer_cnt], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, skip_connect_input=processed_skip_connect_input, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif control_flag[1]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_TS_para_tmp, select_para_tmp = new_darts_dfcnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], dfcnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=dfcnn_TS_activation_fn, KB_param=dfcnn_KB_params[layer_cnt], TS_param=None, conv_param=cnn_TS_params[2*layer_cnt:2*(layer_cnt+1)], select_param=select_params[layer_cnt], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, skip_connect_input=processed_skip_connect_input, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif control_flag[2]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_TS_para_tmp, select_para_tmp = new_darts_dfcnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], dfcnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=dfcnn_TS_activation_fn, KB_param=dfcnn_KB_params[layer_cnt], TS_param=dfcnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], conv_param=cnn_TS_params[2*layer_cnt:2*(layer_cnt+1)], select_param=select_params[layer_cnt], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, skip_connect_input=processed_skip_connect_input, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif control_flag[3]:
                    layer_tmp, KB_para_tmp, TS_para_tmp, cnn_TS_para_tmp, select_para_tmp = new_darts_dfcnn_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], dfcnn_KB_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_sizes[2*layer_cnt:2*(layer_cnt+1)], dfcnn_TS_stride_sizes[2*layer_cnt:2*(layer_cnt+1)], layer_cnt, task_index, activation_fn=activation_fn, para_activation_fn=dfcnn_TS_activation_fn, KB_param=None, TS_param=dfcnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], conv_param=cnn_TS_params[2*layer_cnt:2*(layer_cnt+1)], select_param=select_params[layer_cnt], KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, skip_connect_input=processed_skip_connect_input, use_numpy_var_in_graph=use_numpy_var_in_graph)
            layers.append(layer_tmp)
            layers_for_skip.append(layer_tmp)
            dfcnn_shared_params_return = dfcnn_shared_params_return + KB_para_tmp
            dfcnn_TS_params_return = dfcnn_TS_params_return + TS_para_tmp
            cnn_TS_params_return = cnn_TS_params_return + cnn_TS_para_tmp
            select_params_return = select_params_return + select_para_tmp
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
    return (layers, dfcnn_shared_params_return, dfcnn_TS_params_return, cnn_TS_params_return, select_params_return, output_dim)

def new_darts_dfcnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, dfcnn_KB_sizes, dfcnn_TS_sizes, dfcnn_TS_stride_sizes, cnn_activation_fn=tf.nn.relu, dfcnn_TS_activation_fn=tf.nn.relu, fc_activation_fn=tf.nn.relu, dfcnn_KB_params=None, dfcnn_TS_params=None, cnn_TS_params=None, select_params=None, fc_params=None, KB_reg_type=None, TS_reg_type=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, output_type=None, trainable=True, task_index=0, skip_connections=[], use_numpy_var_in_graph=False):
    cnn_model, dfcnn_shared_params_return, dfcnn_TS_params_return, cnn_TS_params_return, cnn_select_params_return, cnn_output_dim = new_darts_dfcnn_net(net_input, k_sizes, ch_sizes, stride_sizes, dfcnn_KB_sizes, dfcnn_TS_sizes, dfcnn_TS_stride_sizes, activation_fn=cnn_activation_fn, dfcnn_TS_activation_fn=dfcnn_TS_activation_fn, dfcnn_KB_params=dfcnn_KB_params, dfcnn_TS_params=dfcnn_TS_params, cnn_TS_params=cnn_TS_params, select_params=select_params, KB_reg_type=KB_reg_type, TS_reg_type=TS_reg_type, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, trainable=trainable, task_index=task_index, skip_connections=skip_connections, use_numpy_var_in_graph=use_numpy_var_in_graph)

    fc_model, fc_params_return = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, use_numpy_var_in_graph=use_numpy_var_in_graph)
    return (cnn_model+fc_model, dfcnn_shared_params_return, dfcnn_TS_params_return, cnn_TS_params_return, cnn_select_params_return, fc_params_return)
