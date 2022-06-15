import os
import torch
import logging.config
import shutil
import pandas as pd
from bokeh.io import output_file, save, show
from bokeh.plotting import figure
from bokeh.layouts import column
#from bokeh.charts import Line, defaults
#
#defaults.width = 800
#defaults.height = 400
#defaults.tools = 'pan,box_zoom,wheel_zoom,box_select,hover,resize,reset,save'
import torch.nn as nn
import csv
import numpy as np
import math
import copy
from models.binarized_modules import Binarize, BinarizeLinear_inference, BinarizeConv2d_inference


def setup_logging(log_file='log.txt'):
    """Setup logging configuration
    """
    logging.basicConfig(level=logging.DEBUG,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


class ResultsLog(object):

    def __init__(self, path='results.csv', plot_path=None):
        self.path = path
        self.plot_path = plot_path or (self.path + '.html')
        self.figures = []
        self.results = None

    def add(self, **kwargs):
        df = pd.DataFrame([kwargs.values()], columns=kwargs.keys())
        if self.results is None:
            self.results = df
        else:
            self.results = self.results.append(df, ignore_index=True)

    def save(self, title='Training Results'):
        if len(self.figures) > 0:
            if os.path.isfile(self.plot_path):
                os.remove(self.plot_path)
            output_file(self.plot_path, title=title)
            plot = column(*self.figures)
            save(plot)
            self.figures = []
        self.results.to_csv(self.path, index=False, index_label=False)

    def load(self, path=None):
        path = path or self.path
        if os.path.isfile(path):
            self.results.read_csv(path)

    def show(self):
        if len(self.figures) > 0:
            plot = column(*self.figures)
            show(plot)

    #def plot(self, *kargs, **kwargs):
    #    line = Line(data=self.results, *kargs, **kwargs)
    #    self.figures.append(line)

    def image(self, *kargs, **kwargs):
        fig = figure()
        fig.image(*kargs, **kwargs)
        self.figures.append(fig)


def save_checkpoint(state, is_best, path='.', filename='checkpoint.pth.tar', save_all=False):
    filename = os.path.join(path, filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.join(path, 'model_best.pth.tar'))
    if save_all:
        shutil.copyfile(filename, os.path.join(
            path, 'checkpoint_epoch_%s.pth.tar' % state['epoch']))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

__optimizers = {
    'SGD': torch.optim.SGD,
    'ASGD': torch.optim.ASGD,
    'Adam': torch.optim.Adam,
    'Adamax': torch.optim.Adamax,
    'Adagrad': torch.optim.Adagrad,
    'Adadelta': torch.optim.Adadelta,
    'Rprop': torch.optim.Rprop,
    'RMSprop': torch.optim.RMSprop
}


def adjust_optimizer(optimizer, epoch, config):
    """Reconfigures the optimizer according to epoch and config dict"""
    def modify_optimizer(optimizer, setting):
        if 'optimizer' in setting:
            optimizer = __optimizers[setting['optimizer']](
                optimizer.param_groups)
            logging.debug('OPTIMIZER - setting method = %s' %
                          setting['optimizer'])
        for param_group in optimizer.param_groups:
            for key in param_group.keys():
                if key in setting:
                    logging.debug('OPTIMIZER - setting %s = %s' %
                                  (key, setting[key]))
                    param_group[key] = setting[key]
        return optimizer

    if callable(config):
        optimizer = modify_optimizer(optimizer, config(epoch))
    else:
        for e in range(epoch + 1):  # run over all epochs - sticky setting
            if e in config:
                optimizer = modify_optimizer(optimizer, config[e])

    return optimizer


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.float().topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

    # kernel_img = model.features[0][0].kernel.data.clone()
    # kernel_img.add_(-kernel_img.min())
    # kernel_img.mul_(255 / kernel_img.max())
    # save_image(kernel_img, 'kernel%s.jpg' % epoch)
    
    

def layer_type(layer):
    #conv type
    #conv = [type, input channel, output channel, kernel size, padding, stride, output bit]
    #s_conv = [type, input channel, output channel, kernel size, padding, stride, output bit, group]
    #linear = [type, input neurons, output neurons, output bit]
    #pool = [type, kernel size, stride]
    type_list = []
    if isinstance(layer, BinarizeConv2d_inference):
        type_list.append('conv')
        type_list.append(layer.in_channels)
        type_list.append(layer.out_channels)
        type_list.append(layer.kernel_size)
        type_list.append(layer.padding)
        type_list.append(layer.stride)
        type_list.append(1)        
    elif isinstance(layer, BinarizeLinear_inference):
        type_list.append('linear')
        type_list.append(layer.in_features)
        type_list.append(layer.out_features)
        type_list.append(1)
    elif isinstance(layer, nn.MaxPool2d):
        type_list.append('pool')
        type_list.append(layer.kernel_size)
        type_list.append(layer.stride)
    
        
    return type_list
    
# 把pytorch model 轉成list
def model2list(model):
    model_list = []
    weight_aomunt = []
    f = 0
    g = 0
    for i,c in enumerate(model.modules()):
        if i==0 or isinstance(c, nn.ModuleList):
            pass
        else:
            if isinstance(c, nn.Sequential) is not True:
                if isinstance(c, BinarizeConv2d_inference):
                    f += 1
                    # print("Conv Layers %d"%(f))
                    # print(layer_type(c))
                    model_list.append(layer_type(c))
                elif isinstance(c, nn.MaxPool2d):                        
                    f += 1
                    # print("Conv Layers %d"%(f))
                    # print(layer_type(c))
                    model_list.append(layer_type(c))
                elif isinstance(c, BinarizeLinear_inference):
                    g += 1
                    # print("Linear Layers %d"%(g))
                    # print(layer_type(c))
                    model_list.append(layer_type(c))
                        
    return model_list, weight_aomunt
    
    
def model_flow(input_data, a_bit, layer_SA, model_list, p=False):
    tmp_w = input_data.size(1) # feature width
    tmp_h = input_data.size(2) # feature height 
    tmp_b = a_bit # feature bit(activation)
    tmp_mul = [0 for i in range(len(model_list))]
    internal_data_read = [0 for i in range(len(model_list))]
    internal_data_write = [0 for i in range(len(model_list))]
    cal_cycle = [] # 統計每一層conv weight要掃幾次
    cnt = 0
    SA_num = 0
    conv_count = 0
    layer_w = []
    layer_h = []
#    print("input width: %d"%(tmp_w))
#    print("input height: %d"%(tmp_h))
    for i, layer in enumerate(model_list):
        tmp_mul = 0
        tmp_add = 0


        if (layer[0]=='conv'):
            tmp_w = (tmp_w + layer[4][0] * 2 - layer[3][0]) // layer[5][0]
            tmp_h = (tmp_h + layer[4][0] * 2 - layer[3][0]) // layer[5][0]
            layer_w.append(tmp_w)
            SA_num = SA_num + layer_SA[cnt]*tmp_w*tmp_h
            conv_count += 1
            cnt = cnt + 1

        if (layer[0]=='linear' and layer[2]!=10):

            SA_num = SA_num + layer_SA[cnt]
            cnt = cnt + 1

         
        
        elif (layer[0]=='pool'): # pooling
            tmp_w = (tmp_w - layer[1]) // layer[2] + 1
            tmp_h = (tmp_h - layer[1]) // layer[2] + 1
            cal_cycle.append(0)
        
        if p is True:
            print('=====')
            # print("layer %d %s"%(i+1, layer[0]))
            # print("temp width: %d"%(tmp_w))
            # print("temp height: %d"%(tmp_h))
            # print("tmp multiplications: %d(%.3e)"%(tmp_mul[i], tmp_mul[i]))
    if p is True:
        # print("output width: %d"%(tmp_w))
        # print("output height: %d"%(tmp_h))
        print("total multiplications: %d(%.3e)"%(sum(tmp_mul), sum(tmp_mul)))

    return SA_num, layer_w, conv_count
    
    
    
    
    
