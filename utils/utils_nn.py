import numpy as np
import tensorflow as tf

from utils.utils import *



#### function to generate network of cnn->ffnn
def new_cnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_activation_fn=tf.nn.relu, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, skip_connections=[], trainable=True, use_numpy_var_in_graph=False):
    cnn_model, cnn_params, cnn_output_dim = new_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=cnn_activation_fn, params=cnn_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, skip_connections=skip_connections, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)

    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
    return (cnn_model+fc_model, cnn_params, fc_params)


##############################################################################
################# Hard-Parameter Sharing Model
##############################################################################
#### function to generate HPS model of CNN-FC ver. (hard-shared conv layers -> task-dependent fc layers)
def new_hardparam_cnn_fc_nets(net_inputs, k_sizes, ch_sizes, stride_sizes, fc_sizes, num_task, cnn_activation_fn=tf.nn.relu, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, skip_connections=[]):
    num_acc_specific_params, num_specific_params_tmp = [0], 0
    for a in fc_sizes:
        num_specific_params_tmp += 2 * len(a)
        num_acc_specific_params.append(num_specific_params_tmp)

    models, cnn_params_return, fc_params_return = [], [], []
    for task_cnt in range(num_task):
        if task_cnt == 0 and fc_params is None:
            model_tmp, cnn_params_return, fc_param_tmp = new_cnn_fc_net(net_inputs[task_cnt], k_sizes, ch_sizes, stride_sizes, fc_sizes[task_cnt], cnn_activation_fn=cnn_activation_fn, cnn_params=cnn_params, fc_activation_fn=fc_activation_fn, fc_params=None, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, input_size=input_size[0:2], output_type=output_type, skip_connections=list(skip_connections))
        elif task_cnt == 0:
            model_tmp, cnn_params_return, fc_param_tmp = new_cnn_fc_net(net_inputs[task_cnt], k_sizes, ch_sizes, stride_sizes, fc_sizes[task_cnt], cnn_activation_fn=cnn_activation_fn, cnn_params=cnn_params, fc_activation_fn=fc_activation_fn, fc_params=fc_params[task_cnt], padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, input_size=input_size[0:2], output_type=output_type, skip_connections=list(skip_connections))
        elif fc_params is None:
            model_tmp, _, fc_param_tmp = new_cnn_fc_net(net_inputs[task_cnt], k_sizes, ch_sizes, stride_sizes, fc_sizes[task_cnt], cnn_activation_fn=cnn_activation_fn, cnn_params=cnn_params_return, fc_activation_fn=fc_activation_fn, fc_params=None, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, input_size=input_size[0:2], output_type=output_type, skip_connections=list(skip_connections))
        else:
            model_tmp, _, fc_param_tmp = new_cnn_fc_net(net_inputs[task_cnt], k_sizes, ch_sizes, stride_sizes, fc_sizes[task_cnt], cnn_activation_fn=cnn_activation_fn, cnn_params=cnn_params_return, fc_activation_fn=fc_activation_fn, fc_params=fc_params[task_cnt], padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, input_size=input_size[0:2], output_type=output_type, skip_connections=list(skip_connections))

        models.append(model_tmp)
        fc_params_return.append(fc_param_tmp)

    return (models, cnn_params_return, fc_params_return)


def new_flexible_hardparam_cnn_fc_nets(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_sharing, cnn_activation_fn=tf.nn.relu, shared_cnn_params=None, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, skip_connections=[], trainable=True, trainable_shared=True, use_numpy_var_in_graph=False):
    num_conv_layers = [len(k_sizes)//2, len(ch_sizes)-1, len(stride_sizes)//2, len(cnn_sharing)]
    assert (all([num_conv_layers[i]==num_conv_layers[i+1] for i in range(len(num_conv_layers)-1)])), "Given parameters of conv layers don't match each other!"
    num_conv_layers = num_conv_layers[0]

    if cnn_params is None:
        cnn_params = [None for _ in range(2*num_conv_layers)]
    if shared_cnn_params is None:
        shared_cnn_params = [None for _ in range(2*num_conv_layers)]
    if fc_params is None:
        fc_params = [None for _ in range(2*len(fc_sizes))]

    layers_for_skip, next_skip_connect = [net_input], None
    cnn_model, conv_params_to_return, shared_conv_params_to_return = [], [], []
    with tf.name_scope('conv_net'):
        if len(k_sizes) < 2:
            #### for the case that hard-parameter shared network does not have shared layers
            cnn_model.append(net_input)
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
                        if max_pool and (pool_sizes[2*layer_cnt_tmp] > 1 or pool_sizes[2*layer_cnt_tmp+1] > 1):
                            processed_skip_connect_input = tf.nn.max_pool(processed_skip_connect_input, ksize=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], strides=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], padding=padding_type)
                else:
                    processed_skip_connect_input = None

                if layer_cnt == 0 and cnn_sharing[layer_cnt]:
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=shared_cnn_params[2*layer_cnt], bias=shared_cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable_shared, use_numpy_var_in_graph=use_numpy_var_in_graph, name_scope='HPS_conv_layer')
                elif layer_cnt == 0:
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif cnn_sharing[layer_cnt]:
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=cnn_model[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt + 2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=shared_cnn_params[2*layer_cnt], bias=shared_cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable_shared, use_numpy_var_in_graph=use_numpy_var_in_graph, name_scope='HPS_conv_layer')
                else:
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=cnn_model[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt + 2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                cnn_model.append(layer_tmp)
                layers_for_skip.append(layer_tmp)
                conv_params_to_return = conv_params_to_return + para_tmp
                if cnn_sharing[layer_cnt]:
                    shared_conv_params_to_return = shared_conv_params_to_return + para_tmp
                else:
                    shared_conv_params_to_return = shared_conv_params_to_return + [None, None]
                if layer_cnt == skip_connect_out:
                    next_skip_connect = None

        #### flattening output
        output_dim = [int(cnn_model[-1].shape[1] * cnn_model[-1].shape[2] * cnn_model[-1].shape[3])]
        cnn_model.append(tf.reshape(cnn_model[-1], [-1, output_dim[0]]))

        #### add dropout layer
        if dropout:
            cnn_model.append(tf.nn.dropout(cnn_model[-1], dropout_prob))

    ## add fc layers
    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net', trainable=trainable)

    return (cnn_model + fc_model, conv_params_to_return, shared_conv_params_to_return, fc_params)


def new_flexible_hardparam_cnn_fc_nets_ver2(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_sharing, cnn_activation_fn=tf.nn.relu, shared_cnn_params=None, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, skip_connections=[], trainable=True, trainable_shared=True, use_numpy_var_in_graph=False):
    num_conv_layers = [len(k_sizes)//2, len(ch_sizes)-1, len(stride_sizes)//2, len(cnn_sharing)]
    assert (all([num_conv_layers[i]==num_conv_layers[i+1] for i in range(len(num_conv_layers)-1)])), "Given parameters of conv layers don't match each other!"
    num_conv_layers = num_conv_layers[0]

    if cnn_params is None:
        cnn_params = [None for _ in range(2*num_conv_layers)]
    if shared_cnn_params is None:
        shared_cnn_params = [None for _ in range(2*num_conv_layers)]
    if fc_params is None:
        fc_params = [None for _ in range(2*len(fc_sizes))]

    layers_for_skip, next_skip_connect = [net_input], None
    cnn_model, conv_params_to_return, shared_conv_params_to_return = [], [], []
    with tf.name_scope('conv_net'):
        if len(k_sizes) < 2:
            #### for the case that hard-parameter shared network does not have shared layers
            cnn_model.append(net_input)
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
                        if max_pool and (pool_sizes[2*layer_cnt_tmp] > 1 or pool_sizes[2*layer_cnt_tmp+1] > 1):
                            processed_skip_connect_input = tf.nn.max_pool(processed_skip_connect_input, ksize=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], strides=[1]+pool_sizes[2*layer_cnt_tmp:2*(layer_cnt_tmp+1)]+[1], padding=padding_type)
                else:
                    processed_skip_connect_input = None

                if layer_cnt == 0 and cnn_sharing[layer_cnt]:
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=shared_cnn_params[2*layer_cnt], bias=shared_cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable_shared, use_numpy_var_in_graph=use_numpy_var_in_graph, name_scope='HPS_conv_layer')
                elif layer_cnt == 0:
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                elif cnn_sharing[layer_cnt]:
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=cnn_model[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt + 2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=shared_cnn_params[2*layer_cnt], bias=shared_cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable_shared, use_numpy_var_in_graph=use_numpy_var_in_graph, name_scope='HPS_conv_layer')
                else:
                    layer_tmp, para_tmp = new_cnn_layer(layer_input=cnn_model[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt + 2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
                cnn_model.append(layer_tmp)
                layers_for_skip.append(layer_tmp)
                if cnn_sharing[layer_cnt]:
                    shared_conv_params_to_return = shared_conv_params_to_return + para_tmp
                    conv_params_to_return = conv_params_to_return + [None, None]
                else:
                    shared_conv_params_to_return = shared_conv_params_to_return + [None, None]
                    conv_params_to_return = conv_params_to_return + para_tmp
                if layer_cnt == skip_connect_out:
                    next_skip_connect = None

        #### flattening output
        output_dim = [int(cnn_model[-1].shape[1] * cnn_model[-1].shape[2] * cnn_model[-1].shape[3])]
        cnn_model.append(tf.reshape(cnn_model[-1], [-1, output_dim[0]]))

        #### add dropout layer
        if dropout:
            cnn_model.append(tf.nn.dropout(cnn_model[-1], dropout_prob))

    ## add fc layers
    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, tensorboard_name_scope='fc_net', trainable=trainable)

    return (cnn_model + fc_model, conv_params_to_return, shared_conv_params_to_return, fc_params)



##############################################################################
################# Tensor Factored Model
##############################################################################
######## Lifelong Learning - based on Adrian Bulat, et al. Incremental Multi-domain Learning with Network Latent Tensor Factorization
def new_TF_KB_param(shape, layer_number, init_tensor=None, trainable=True):
    kb_name = 'KB_'+str(layer_number)
    if init_tensor is None:
        param_to_return = tf.get_variable(name=kb_name, shape=shape, dtype=tf.float32, trainable=trainable)
    elif type(init_tensor) == np.ndarray:
        param_to_return = tf.get_variable(name=kb_name, shape=shape, dtype=tf.float32, initializer=tf.constant_initializer(init_tensor), trainable=trainable)
    else:
        param_to_return = init_tensor
    return param_to_return

def new_TF_TS_param(shape, layer_number, task_number, init_tensor, trainable):
    params_name = ['TF_Wch0_'+str(layer_number)+'_'+str(task_number), 'TF_Wch1_'+str(layer_number)+'_'+str(task_number), 'TF_Wch2_'+str(layer_number)+'_'+str(task_number), 'TF_Wch3_'+str(layer_number)+'_'+str(task_number), 'b_'+str(layer_number)+'_'+str(task_number)]
    params_to_return = []
    for i, (t, n) in enumerate(zip(init_tensor, params_name)):
        if t is None:
            params_to_return.append(tf.get_variable(name=n, shape=shape[i], dtype=tf.float32, trainable=trainable))
        elif type(t) == np.ndarray:
            params_to_return.append(tf.get_variable(name=n, shape=shape[i], dtype=tf.float32, trainable=trainable, initializer=tf.constant_initializer(t)))
        else:
            params_to_return.append(t)
    return params_to_return


def new_tensorfactored_conv_layer(layer_input, k_size, ch_size, stride_size, layer_num, task_num, activation_fn=tf.nn.relu, KB_param=None, TS_param=None, padding_type='SAME', max_pool=False, pool_size=None, skip_connect_input=None, highway_connect_type=0, highway_W=None, highway_b=None, trainable=True, trainable_KB=True):
    with tf.name_scope('TF_conv'):
        KB_param = new_TF_KB_param(k_size+ch_size, layer_num, KB_param, trainable=trainable_KB)

        TS_param = new_TF_TS_param([[a, a] for a in k_size+ch_size]+[ch_size[1]], layer_num, task_num, TS_param if TS_param else [None, None, None, None, None], trainable)

    with tf.name_scope('TF_param_gen'):
        W = KB_param
        for t in TS_param[:-1]:
            W = tf.tensordot(W, t, [[0], [0]])
        b = TS_param[-1]

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


def new_hybrid_tensorfactored_cnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_sharing, cnn_activation_fn=tf.nn.relu, cnn_KB_params=None, cnn_TS_params=None, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, task_index=0, skip_connections=[], highway_connect_type=0, cnn_highway_params=None, trainable=True, trainable_KB=True):
    _num_TS_param_per_layer = 5

    num_conv_layers = [len(k_sizes)//2, len(ch_sizes)-1, len(stride_sizes)//2, len(cnn_sharing)]
    assert (all([(num_conv_layers[i]==num_conv_layers[i+1]) for i in range(len(num_conv_layers)-1)])), "Parameters related to conv layers are wrong!"
    num_conv_layers = num_conv_layers[0]

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
    with tf.name_scope('Hybrid_TensorFactorized_CNN'):
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

            if control_flag[0] and cnn_sharing[layer_cnt]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_tensorfactored_conv_layer(net_input if layer_cnt<1 else cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], layer_cnt, task_index, activation_fn=cnn_activation_fn, KB_param=None, TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
            elif control_flag[1] and cnn_sharing[layer_cnt]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_tensorfactored_conv_layer(net_input if layer_cnt<1 else cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], layer_cnt, task_index, activation_fn=cnn_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=None, padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
            elif control_flag[2] and cnn_sharing[layer_cnt]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_tensorfactored_conv_layer(net_input if layer_cnt<1 else cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], layer_cnt, task_index, activation_fn=cnn_activation_fn, KB_param=cnn_KB_params[layer_cnt], TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
            elif control_flag[3] and cnn_sharing[layer_cnt]:
                layer_tmp, KB_para_tmp, TS_para_tmp, cnn_gen_para_tmp, highway_para_tmp = new_tensorfactored_conv_layer(net_input if layer_cnt<1 else cnn_model[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)], ch_sizes[layer_cnt:layer_cnt+2], [1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], layer_cnt, task_index, activation_fn=cnn_activation_fn, KB_param=None, TS_param=cnn_TS_params[_num_TS_param_per_layer*layer_cnt:_num_TS_param_per_layer*(layer_cnt+1)], padding_type=padding_type, max_pool=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, highway_connect_type=highway_connect_type, highway_W=highway_para_tmp[0], highway_b=highway_para_tmp[1], trainable=trainable, trainable_KB=trainable_KB)
            elif (not cnn_sharing[layer_cnt]):
                layer_tmp, para_tmp = new_cnn_layer(layer_input=net_input if layer_cnt<1 else cnn_model[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=cnn_activation_fn, weight=cnn_params[2*layer_cnt], bias=cnn_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], skip_connect_input=processed_skip_connect_input, trainable=trainable)

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

    return (cnn_model+fc_model, cnn_KB_to_return, cnn_TS_to_return, cnn_gen_params, cnn_params_to_return, cnn_highway_params_to_return, fc_params)



##############################################################################
################# Progressive Neural Net model
##############################################################################

#### function to add progressive convolution layer
# k_size : [h, w, cin, cout]
def new_progressive_cnn_layer(layer_input, k_size, stride_size=[1, 1, 1, 1], activation_fn=tf.nn.relu, weight=None, bias=None, padding_type='SAME', max_pooling=False, pool_size=None, prev_column_inputs=None, num_prev_cols=1, lat_connect_param=None, trainable=True, dim_reduction_scale=1.0, skip_connect_input=None):
    with tf.name_scope('prog_conv_layer'):
        if weight is None:
            weight = new_weight(shape=k_size, trainable=trainable)
        if bias is None:
            bias = new_bias(shape=[k_size[-1]], trainable=trainable)

        conv_layer = tf.nn.conv2d(layer_input, weight, strides=stride_size, padding=padding_type) + bias

        exist_prev_column = (None not in prev_column_inputs) if (type(prev_column_inputs) == list) else (prev_column_inputs is not None)

        if exist_prev_column:
            exist_lat_param = (not (all([x is None for x in lat_connect_param]))) if (type(lat_connect_param) == list) else (lat_connect_param is not None)
            if not exist_lat_param:
                ## generate new params for lateral connection (alpha, (Wv, bv), Wu)
                lat_connect_param, nic = [], int(k_size[2] / (dim_reduction_scale * (num_prev_cols+1.0)))
                for col_cnt in range(num_prev_cols):
                    lat_connect_param = lat_connect_param + [new_weight(shape=[1], trainable=trainable), new_weight(shape=[1, 1, k_size[2], nic], trainable=trainable), new_bias(shape=[nic], trainable=trainable), new_weight(shape=k_size[0:2]+[nic, k_size[3]], trainable=trainable)]

            ## generate lateral connection
            lateral_outputs = []
            for col_cnt in range(num_prev_cols):
                lat_col_hid1 = tf.multiply(lat_connect_param[4*col_cnt], prev_column_inputs[col_cnt])

                ## dim reduction using conv (k:[1, 1], stride:[1, 1], act:ReLu)
                lat_col_hid2 = tf.nn.relu(tf.nn.conv2d(lat_col_hid1, lat_connect_param[4*col_cnt+1], strides=[1, 1, 1, 1], padding=padding_type) + lat_connect_param[4*col_cnt+2])

                ## conv lateral connection
                lateral_outputs.append(tf.nn.conv2d(lat_col_hid2, lat_connect_param[4*col_cnt+3], strides=stride_size, padding=padding_type))

            conv_layer = conv_layer + tf.reduce_sum(lateral_outputs, axis=0)
        else:
            lat_connect_param = [None for _ in range(4*num_prev_cols)]

        if skip_connect_input is not None:
            shape1, shape2 = conv_layer.get_shape().as_list(), skip_connect_input.get_shape().as_list()
            assert (len(shape1) == len(shape2)), "Shape of layer's output and input of skip connection do not match!"
            assert (all([(x==y) for (x, y) in zip(shape1, shape2)])), "Shape of layer's output and input of skip connection do NOT match!"
            conv_layer = conv_layer + skip_connect_input

        if activation_fn is not None:
            act_conv_layer = activation_fn(conv_layer)

        if max_pooling:
            layer = tf.nn.max_pool(act_conv_layer, ksize=pool_size, strides=pool_size, padding=padding_type)
        else:
            layer = act_conv_layer
    return (layer, [weight, bias], lat_connect_param)


#### function to generate network of progressive convolutional layers
####      prog_conv-pool-prog_conv-pool-...-prog_conv-pool-flat-dropout
####      k_sizes/stride_size/pool_sizes : [x_0, y_0, x_1, y_1, ..., x_m, y_m]
####      ch_sizes : [img_ch, ch_0, ch_1, ..., ch_m]
def new_progressive_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=tf.nn.relu, params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0], prev_column_net=None, lat_connect_params=None, trainable=True, dim_reduction_scale=1.0, skip_connections=[]):
    num_layers = len(k_sizes)//2
    assert (num_layers == len(ch_sizes)-1), "Check the number of progressive cnn layers"

    if not max_pool:
        pool_sizes = [None for _ in range(len(k_sizes))]

    if prev_column_net is None:
        prev_column_net = [[None for _ in range(num_layers)]]
    num_prev_nets = len(prev_column_net)
    lat_param_cnt_multiplier = 4*num_prev_nets

    if lat_connect_params is None:
        lat_connect_params = [None for _ in range(lat_param_cnt_multiplier*num_layers)]

    layers_for_skip, next_skip_connect = [net_input], None
    with tf.name_scope('prog_conv_net'):
        if num_layers < 1:
            #### for the case that hard-parameter shared network does not have shared layers
            return (net_input, [])
        elif params is None:
            #### network & parameters are new
            layers, params, lat_params = [], [], []
            for layer_cnt in range(num_layers):
                next_skip_connect = skip_connections.pop(0) if (len(skip_connections) > 0 and next_skip_connect is None) else next_skip_connect
                if next_skip_connect is not None:
                    skip_connect_in, skip_connect_out = next_skip_connect
                    assert (skip_connect_in > -1 and skip_connect_out > -1), "Given skip connection has error (try connecting non-existing layer)"
                else:
                    skip_connect_in, skip_connect_out = -1, -1

                if layer_cnt == 0:
                    layer_tmp, para_tmp, lat_para_tmp = new_progressive_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], num_prev_cols=num_prev_nets, trainable=trainable, dim_reduction_scale=dim_reduction_scale, skip_connect_input=layers_for_skip[skip_connect_in] if (layer_cnt==skip_connect_out) else None)
                else:
                    layer_tmp, para_tmp, lat_para_tmp = new_progressive_cnn_layer(layer_input=layers[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], prev_column_inputs=[prev_column_net[c][layer_cnt-1] for c in range(num_prev_nets)], num_prev_cols=num_prev_nets, lat_connect_param=lat_connect_params[lat_param_cnt_multiplier*layer_cnt:lat_param_cnt_multiplier*(layer_cnt+1)], trainable=trainable, dim_reduction_scale=dim_reduction_scale, skip_connect_input=layers_for_skip[skip_connect_in] if (layer_cnt==skip_connect_out) else None)
                layers.append(layer_tmp)
                layers_for_skip.append(layer_tmp)
                params = params + para_tmp
                lat_params = lat_params + lat_para_tmp
        else:
            #### network generated from existing parameters
            layers, lat_params = [], lat_connect_params
            for layer_cnt in range(num_layers):
                next_skip_connect = skip_connections.pop(0) if (len(skip_connections) > 0 and next_skip_connect is None) else next_skip_connect
                if next_skip_connect is not None:
                    skip_connect_in, skip_connect_out = next_skip_connect
                    assert (skip_connect_in > -1 and skip_connect_out > -1), "Given skip connection has error (try connecting non-existing layer)"
                else:
                    skip_connect_in, skip_connect_out = -1, -1

                if layer_cnt == 0:
                    layer_tmp, _, _ = new_progressive_cnn_layer(layer_input=net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, weight=params[2*layer_cnt], bias=params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], num_prev_cols=num_prev_nets, trainable=trainable, dim_reduction_scale=dim_reduction_scale, skip_connect_input=layers_for_skip[skip_connect_in] if (layer_cnt==skip_connect_out) else None)
                else:
                    layer_tmp, _, _ = new_progressive_cnn_layer(layer_input=layers[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, weight=params[2*layer_cnt], bias=params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], prev_column_inputs=[prev_column_net[c][layer_cnt-1] for c in range(num_prev_nets)], num_prev_cols=num_prev_nets, lat_connect_param=lat_connect_params[lat_param_cnt_multiplier*layer_cnt:lat_param_cnt_multiplier*(layer_cnt+1)], trainable=trainable, dim_reduction_scale=dim_reduction_scale, skip_connect_input=layers_for_skip[skip_connect_in] if (layer_cnt==skip_connect_out) else None)
                layers.append(layer_tmp)
                layers_for_skip.append(layer_tmp)

        #### flattening output
        if flat_output:
            output_dim = [int(layers[-1].shape[1]*layers[-1].shape[2]*layers[-1].shape[3])]
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))
        else:
            output_dim = layers[-1].shape[1:]

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, params, lat_params, output_dim)


#### function to generate network of progressive cnn->ffnn
def new_progressive_cnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_activation_fn=tf.nn.relu, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, prev_net=None, cnn_lateral_params=None, trainable=True, dim_reduction_scale=1.0, skip_connections=[], use_numpy_var_in_graph=False):
    cnn_model, cnn_params, cnn_lateral_params, cnn_output_dim = new_progressive_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=cnn_activation_fn, params=cnn_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, prev_column_net=prev_net, lat_connect_params=cnn_lateral_params, trainable=trainable, dim_reduction_scale=dim_reduction_scale, skip_connections=skip_connections)

    fc_model, fc_params = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, use_numpy_var_in_graph=use_numpy_var_in_graph)
    return (cnn_model+fc_model, cnn_params, cnn_lateral_params, fc_params)



#### function to generate DARTS-based network for selective sharing on multi-task CNN (or Hard-parameter Sharing)
def new_darts_conv_layer(layer_input, k_size, stride_size=[1, 1, 1, 1], activation_fn=tf.nn.relu, shared_weight=None, shared_bias=None, TS_weight=None, TS_bias=None, select_param=None, padding_type='SAME', max_pooling=False, pool_size=None, trainable=True, skip_connect_input=None, name_scope='darts_conv_layer', use_numpy_var_in_graph=False):
    with tf.name_scope(name_scope):
        ## init shared params
        if shared_weight is None:
            shared_weight = new_weight(shape=k_size, trainable=trainable)
        elif (type(shared_weight) == np.ndarray) and not use_numpy_var_in_graph:
            shared_weight = new_weight(shape=k_size, init_tensor=shared_weight, trainable=trainable)
        if shared_bias is None:
            shared_bias = new_bias(shape=[k_size[-1]], trainable=trainable)
        elif (type(shared_bias) == np.ndarray) and not use_numpy_var_in_graph:
            shared_bias = new_bias(shape=[k_size[-1]], init_tensor=shared_bias, trainable=trainable)
        ## init task-specific params
        if TS_weight is None:
            TS_weight = new_weight(shape=k_size, trainable=trainable)
        elif (type(TS_weight) == np.ndarray) and not use_numpy_var_in_graph:
            TS_weight = new_weight(shape=k_size, init_tensor=TS_weight, trainable=trainable)
        if TS_bias is None:
            TS_bias = new_bias(shape=[k_size[-1]], trainable=trainable)
        elif (type(TS_bias) == np.ndarray) and not use_numpy_var_in_graph:
            TS_bias = new_bias(shape=[k_size[-1]], init_tensor=TS_bias, trainable=trainable)
        ## init DARTS-selection params
        if select_param is None:
            select_param = new_weight(shape=[2], init_tensor=np.zeros(2, dtype=np.float32), trainable=trainable)
        elif (type(select_param) == np.ndarray) and not use_numpy_var_in_graph:
            select_param = new_weight(shape=[2], init_tensor=select_param, trainable=trainable)

        mixing_weight = tf.reshape(tf.nn.softmax(select_param), [2,1])
        shared_conv_layer = tf.nn.conv2d(layer_input, shared_weight, strides=stride_size, padding=padding_type) + shared_bias
        TS_conv_layer = tf.nn.conv2d(layer_input, TS_weight, strides=stride_size, padding=padding_type) + TS_bias

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
    return (layer, [shared_weight, shared_bias], [TS_weight, TS_bias], [select_param])

def new_darts_conv_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=tf.nn.relu, shared_params=None, TS_params=None, select_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0], trainable=True, skip_connections=[], use_numpy_var_in_graph=False):
    if not max_pool:
        pool_sizes = [None for _ in range(len(k_sizes))]

    if shared_params is None:
        shared_params = [None for _ in range(len(k_sizes))]
    if TS_params is None:
        TS_params = [None for _ in range(len(k_sizes))]
    if select_params is None:
        select_params = [None for _ in range(len(k_sizes)//2)]

    layers_for_skip, next_skip_connect = [net_input], None
    layers, shared_params_to_return, TS_params_to_return, select_params_to_return = [], [], [], []
    with tf.name_scope('DARTS_conv_net'):
        for layer_cnt in range(len(k_sizes)//2):
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
                layer_tmp, shared_para_tmp, TS_para_tmp, select_para_tmp = new_darts_conv_layer(net_input, k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, shared_weight=shared_params[2*layer_cnt], shared_bias=shared_params[2*layer_cnt+1], TS_weight=TS_params[2*layer_cnt], TS_bias=TS_params[2*layer_cnt+1], select_param=select_params[layer_cnt], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, skip_connect_input=processed_skip_connect_input, use_numpy_var_in_graph=use_numpy_var_in_graph)
            else:
                layer_tmp, shared_para_tmp, TS_para_tmp, select_para_tmp = new_darts_conv_layer(layers[layer_cnt-1], k_size=k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, shared_weight=shared_params[2*layer_cnt], shared_bias=shared_params[2*layer_cnt+1], TS_weight=TS_params[2*layer_cnt], TS_bias=TS_params[2*layer_cnt+1], select_param=select_params[layer_cnt], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, skip_connect_input=processed_skip_connect_input, use_numpy_var_in_graph=use_numpy_var_in_graph)
            layers.append(layer_tmp)
            layers_for_skip.append(layer_tmp)
            shared_params_to_return = shared_params_to_return + shared_para_tmp
            TS_params_to_return = TS_params_to_return + TS_para_tmp
            select_params_to_return = select_params_to_return + select_para_tmp
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
    return (layers, shared_params_to_return, TS_params_to_return, select_params_to_return, output_dim)


def new_darts_cnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_activation_fn=tf.nn.relu, cnn_shared_params=None, cnn_TS_params=None, select_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, trainable=True, skip_connections=[], use_numpy_var_in_graph=False):
    cnn_model, cnn_shared_params_return, cnn_TS_params_return, cnn_select_params_return, cnn_output_dim = new_darts_conv_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=cnn_activation_fn, shared_params=cnn_shared_params, TS_params=cnn_TS_params, select_params=select_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, trainable=trainable, skip_connections=skip_connections)

    fc_model, fc_params_return = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, use_numpy_var_in_graph=use_numpy_var_in_graph)
    return (cnn_model+fc_model, cnn_shared_params_return, cnn_TS_params_return, cnn_select_params_return, fc_params_return)





#### function to generate network of fully-connected layers for Actor-Critic
####      'dim_layers' contains input/output layer
def new_fc_net_for_actor_critic(net_input, dim_layers, activation_fn=tf.nn.relu, params=None, tensorboard_name_scope='actor_critic_fc_net', trainable=True, use_numpy_var_in_graph=False):
    if params is None:
        params = [None for _ in range(2*len(dim_layers)+2)]
    else:
        assert (len(params)==2*len(dim_layers)+2), "The number of given parameters doesn't match the number of required layers!"

    layers, params_to_return = [], []
    assert (len(dim_layers) > 0), "Fully-connected network for Actor-Critic requires at least one layer!"

    with tf.name_scope(tensorboard_name_scope):
        for cnt in range(len(dim_layers)):
            if cnt == 0:
                layer_tmp, para_tmp = new_fc_layer(net_input, dim_layers[cnt], activation_fn=activation_fn, weight=params[0], bias=params[1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
            elif cnt == len(dim_layers)-1:
                with tf.name_scope('pi'):
                    layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], activation_fn=None, weight=params[2*cnt], bias=params[2*cnt+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
            else:
                layer_tmp, para_tmp = new_fc_layer(layers[cnt-1], dim_layers[cnt], activation_fn=activation_fn, weight=params[2*cnt], bias=params[2*cnt+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
            layers.append(layer_tmp)
            params_to_return = params_to_return + para_tmp
        with tf.name_scope('value'):
            value_layer_input = net_input if len(dim_layers) < 2 else layers[len(dim_layers)-2]
            layer_tmp, para_tmp = new_fc_layer(value_layer_input, 1, activation_fn=None, weight=params[2*len(dim_layers)], bias=params[2*len(dim_layers)+1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
        layers.append(layer_tmp)
        params_to_return = params_to_return + para_tmp
    return (layers, params_to_return)


#### function to generate network for Actor-Critic (only last (fully-connected) layer is split into policy and actor
def new_cnn_actor_critic_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_activation_fn=tf.nn.relu, cnn_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], skip_connections=[], trainable=True, use_numpy_var_in_graph=False):
    cnn_model, cnn_params, cnn_output_dim = new_cnn_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=cnn_activation_fn, params=cnn_params, padding_type=padding_type, max_pool=max_pool, pool_sizes=pool_sizes, dropout=dropout, dropout_prob=dropout_prob, flat_output=True, input_size=input_size, skip_connections=skip_connections, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)

    fc_model, fc_params = new_fc_net_for_actor_critic(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
    return (cnn_model+fc_model, cnn_params, fc_params)



#### function to generate layer of conditional channel-gated network
def new_channel_gated_conv_layer(layer_input, k_size, gating_module_size, gating_temperature, stride_size=[1, 1, 1, 1], activation_fn=tf.nn.relu, activation_fn_gating=tf.nn.relu, conv_weight=None, conv_bias=None, gating_weights=None, padding_type='SAME', max_pooling=False, pool_size=None, trainable=True, name_scope='chgated_conv_layer', use_numpy_var_in_graph=False):
    batch_size = int(layer_input.get_shape()[0])
    with tf.name_scope(name_scope):
        ## base conv layer
        conv_output, conv_params = new_cnn_layer(layer_input, k_size, stride_size, activation_fn, weight=conv_weight, bias=conv_bias, padding_type=padding_type, max_pooling=max_pooling, pool_size=pool_size, skip_connect_input=None, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)

        ## gating module
        flattened_input, gating_sizes = tf.reshape(layer_input, [batch_size, -1]), list(gating_module_size)+[k_size[-1]]
        gating_layers, gating_params = new_fc_net(flattened_input, gating_sizes, activation_fn=activation_fn_gating, params=gating_weights, output_type=None, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
        channelwise_gate = tf.nn.softmax(gating_layers[-1]/gating_temperature)

        ## channel-wise gating
        temp_channelwise_gate = tf.expand_dims(tf.expand_dims(channelwise_gate, axis=1), axis=2)
        layer_output = tf.multiply(conv_output, temp_channelwise_gate)
    return (layer_output, channelwise_gate, conv_params, gating_params)

def new_channel_gated_conv_net(net_input, k_sizes, ch_sizes, gating_module_sizes, gating_temperature, stride_sizes, activation_fn=tf.nn.relu, activate_fn_gating=tf.nn.relu, conv_params=None, gating_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0], trainable=True, use_numpy_var_in_graph=False):
    num_layers, num_gating_layers = len(k_sizes)//2, len(gating_module_sizes)+1

    if not max_pool:
        pool_sizes = [None for _ in range(2*num_layers)]

    if conv_params is None:
        conv_params = [None for _ in range(2*num_layers)]
    if gating_params is None:
        gating_params = [None for _ in range(2*num_gating_layers*num_layers)]

    layers, gates, conv_params_to_return, gating_params_to_return = [], [], [], []
    with tf.name_scope('ChannelGated_conv_net'):
        for layer_cnt in range(num_layers):
            if layer_cnt == 0:
                layer_tmp, gate_tmp, conv_para_tmp, gating_para_tmp = new_channel_gated_conv_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], gating_module_sizes, gating_temperature, stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, activation_fn_gating=activate_fn_gating, conv_weight=conv_params[2*layer_cnt], conv_bias=conv_params[2*layer_cnt+1], gating_weights=gating_params[2*num_gating_layers*layer_cnt:2*num_gating_layers*(layer_cnt+1)], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
            else:
                layer_tmp, gate_tmp, conv_para_tmp, gating_para_tmp = new_channel_gated_conv_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], gating_module_sizes, gating_temperature, stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, activation_fn_gating=activate_fn_gating, conv_weight=conv_params[2*layer_cnt], conv_bias=conv_params[2*layer_cnt+1], gating_weights=gating_params[2*num_gating_layers*layer_cnt:2*num_gating_layers*(layer_cnt+1)], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
            layers.append(layer_tmp)
            gates.append(gate_tmp)
            conv_params_to_return = conv_params_to_return + conv_para_tmp
            gating_params_to_return = gating_params_to_return + gating_para_tmp

        #### flattening output
        if flat_output:
            output_dim = [int(layers[-1].shape[1]*layers[-1].shape[2]*layers[-1].shape[3])]
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))
        else:
            output_dim = layers[-1].shape[1:]

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, gates, conv_params_to_return, gating_params_to_return, output_dim)

def new_channel_gated_cnn_fc_net(net_input, k_sizes, ch_sizes, gating_module_sizes, gating_temperature, stride_sizes, fc_sizes, cnn_activation_fn=tf.nn.relu, gating_activate_fn=tf.nn.relu, conv_params=None, gating_params=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, trainable=True, use_numpy_var_in_graph=False):
    cnn_model, cnn_gates, cnn_params_return, gating_params_return, cnn_output_dim = new_channel_gated_conv_net(net_input, k_sizes, ch_sizes, gating_module_sizes, gating_temperature, stride_sizes, cnn_activation_fn, gating_activate_fn, conv_params, gating_params, padding_type, max_pool, pool_sizes, dropout, dropout_prob, flat_output=True, input_size=input_size, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)

    fc_model, fc_params_return = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, use_numpy_var_in_graph=use_numpy_var_in_graph, trainable=trainable)
    return (cnn_model+fc_model, cnn_gates, cnn_params_return, gating_params_return, fc_params_return)


#### function to generate a conv layer based on additive parameter decomposition
def new_APD_conv_layer(layer_input, k_size, stride_size=[1, 1, 1, 1], activation_fn=tf.nn.relu, shared_weight=None, mask_weight=None, task_adaptive_weight=None, conv_bias=None, padding_type='SAME', max_pooling=False, pool_size=None, trainable=True, name_scope='chgated_conv_layer', use_numpy_var_in_graph=False):
    batch_size = int(layer_input.get_shape()[0])
    with tf.name_scope(name_scope):
        ## shared weight
        if shared_weight is None or (type(shared_weight) == np.ndarray and not use_numpy_var_in_graph):
            shared_weight = new_weight(shape=k_size, init_tensor=shared_weight, trainable=trainable)
        ## mask and task-adaptive parameter
        if mask_weight is None or (type(mask_weight) == np.ndarray and not use_numpy_var_in_graph):
            mask_weight = new_weight(shape=k_size, init_tensor=mask_weight, trainable=trainable)
        if task_adaptive_weight is None or (type(task_adaptive_weight) == np.ndarray and not use_numpy_var_in_graph):
            task_adaptive_weight = new_weight(shape=k_size, init_tensor=task_adaptive_weight, trainable=trainable)
        if conv_bias is None or (type(conv_bias) == np.ndarray and not use_numpy_var_in_graph):
            conv_bias = new_bias(shape=[k_size[-1]], init_tensor=conv_bias, trainable=trainable)

        ## conv with shared & mask & task-adaptive parameter
        conv_weight = tf.multiply(shared_weight, tf.nn.sigmoid(mask_weight)) + task_adaptive_weight
        conv_output, conv_params = new_cnn_layer(layer_input, k_size, stride_size, activation_fn, weight=conv_weight, bias=conv_bias, padding_type=padding_type, max_pooling=max_pooling, pool_size=pool_size, skip_connect_input=None, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
    return (conv_output, [shared_weight, conv_bias], mask_weight, task_adaptive_weight)

def new_APD_conv_net(net_input, k_sizes, ch_sizes, stride_sizes, activation_fn=tf.nn.relu, conv_params=None, mask_weights=None, task_adaptive_weights=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, flat_output=False, input_size=[0, 0], trainable=True, use_numpy_var_in_graph=False):
    num_layers = len(k_sizes)//2

    if not max_pool:
        pool_sizes = [None for _ in range(2*num_layers)]

    if conv_params is None:
        conv_params = [None for _ in range(2*num_layers)]
    if mask_weights is None:
        mask_weights = [None for _ in range(num_layers)]
    if task_adaptive_weights is None:
        task_adaptive_weights = [None for _ in range(num_layers)]

    layers, conv_params_to_return, mask_params_to_return, task_adaptive_params_to_return = [], [], [], []
    with tf.name_scope('APD_conv_net'):
        for layer_cnt in range(num_layers):
            if layer_cnt == 0:
                layer_tmp, conv_para_tmp, mask_para_tmp, task_apd_para_tmp = new_APD_conv_layer(net_input, k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, shared_weight=conv_params[2*layer_cnt], mask_weight=mask_weights[layer_cnt], task_adaptive_weight=task_adaptive_weights[layer_cnt], conv_bias=conv_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
            else:
                layer_tmp, conv_para_tmp, mask_para_tmp, task_apd_para_tmp = new_APD_conv_layer(layers[layer_cnt-1], k_sizes[2*layer_cnt:2*(layer_cnt+1)]+ch_sizes[layer_cnt:layer_cnt+2], stride_size=[1]+stride_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], activation_fn=activation_fn, shared_weight=conv_params[2*layer_cnt], mask_weight=mask_weights[layer_cnt], task_adaptive_weight=task_adaptive_weights[layer_cnt], conv_bias=conv_params[2*layer_cnt+1], padding_type=padding_type, max_pooling=max_pool, pool_size=[1]+pool_sizes[2*layer_cnt:2*(layer_cnt+1)]+[1], trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)
            layers.append(layer_tmp)
            conv_params_to_return = conv_params_to_return + conv_para_tmp
            mask_params_to_return.append(mask_para_tmp)
            task_adaptive_params_to_return.append(task_apd_para_tmp)

        #### flattening output
        if flat_output:
            output_dim = [int(layers[-1].shape[1]*layers[-1].shape[2]*layers[-1].shape[3])]
            layers.append(tf.reshape(layers[-1], [-1, output_dim[0]]))
        else:
            output_dim = layers[-1].shape[1:]

        #### add dropout layer
        if dropout:
            layers.append(tf.nn.dropout(layers[-1], dropout_prob))
    return (layers, conv_params_to_return, mask_params_to_return, task_adaptive_params_to_return, output_dim)

def new_APD_cnn_fc_net(net_input, k_sizes, ch_sizes, stride_sizes, fc_sizes, cnn_activation_fn=tf.nn.relu, conv_params=None, mask_weights=None, task_adaptive_weights=None, fc_activation_fn=tf.nn.relu, fc_params=None, padding_type='SAME', max_pool=False, pool_sizes=None, dropout=False, dropout_prob=None, input_size=[0, 0], output_type=None, trainable=True, use_numpy_var_in_graph=False):
    cnn_model, cnn_params_return, mask_params_return, task_adaptive_params_return, cnn_output_dim = new_APD_conv_net(net_input, k_sizes, ch_sizes, stride_sizes, cnn_activation_fn, conv_params, mask_weights, task_adaptive_weights, padding_type, max_pool, pool_sizes, dropout, dropout_prob, flat_output=True, input_size=input_size, trainable=trainable, use_numpy_var_in_graph=use_numpy_var_in_graph)

    fc_model, fc_params_return = new_fc_net(cnn_model[-1], fc_sizes, activation_fn=fc_activation_fn, params=fc_params, output_type=output_type, use_numpy_var_in_graph=use_numpy_var_in_graph, trainable=trainable)
    return (cnn_model+fc_model, cnn_params_return, mask_params_return, task_adaptive_params_return, fc_params_return)