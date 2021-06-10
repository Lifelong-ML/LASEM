import tensorflow as tf
import numpy as np
import re, random, collections
from collections import defaultdict
from numpy import linalg as LA
from sklearn.metrics import roc_curve, auc

from os import getcwd, listdir, mkdir
import scipy.io as spio

#from tensorflow.contrib.memory_stats.python.ops import memory_stats_ops

from utils.utils import convert_array_to_oneHot

_num_tasks = 10


def save_param_to_mat(param_dict):
    param_to_save_format = {}
    for key, val in param_dict.items():
        scope_name = key.split(':')[0]
        scope = scope_name.split('/')[0]
        name = scope_name.split('/')[1]
        new_scope_name = scope + '_' + name
        param_to_save_format[new_scope_name] = val
    return param_to_save_format


def accuracy(preds, labels):
    return (100.0 * np.sum(np.argmax(preds, 1) == np.argmax(labels, 1)) / preds.shape[0])

def RMSE(p, y):
    N = p.shape[0]
    diff = p - y
    return np.sqrt((diff**2).mean())

def ROC_AUC(p, y):
    fpr, tpr, th = roc_curve(y, p)
    _auc = auc(fpr, tpr)
    _roc = (fpr, tpr)
    return _roc, _auc

class CNN_FC_DEN(object):
    def __init__(self, model_hyperpara, train_hyperpara, data_info):
        raise NotImplementedError
