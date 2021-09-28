import os
import torch
import numpy as np
from tools.general import plot_utils, io_utils, log_utils
from tools.ai import torch_utils
import shutil

if __name__ == '__main__':
    ###################################################################################
    # 1. Arguments
    ###################################################################################
    parser = io_utils.Parser()

    # 1. dataset
    parser.add('seed', 1, int)
    
    parser.add('train_domain', 'train', str)
    parser.add('valid_domain', 'valid', str)

    parser.add('root_dir', './../data/flower_split/', str)

    # for folder_names
    parser.add('folder_names', 'daisy,dandelion,roses,sunflowers,tulips', str)
    
    # 2. networks
    parser.add('backbone', 'mobilenetv2', str)
    
    # 3. hyperparameters
    parser.add('image_size', 64, int)
    parser.add('batch_size', 32, int)

    parser.add('tag', 'MobilenetV2', str)

    # 4. training
    parser.add('reset', False, bool)

    parser.add('max_epoch', 100, int)
    parser.add('valid_ratio', 5.0, float)

    parser.add('loss', 'cross_entropy_loss', str)
    
    parser.add('lr', 0.001, float)
    
    parser.add('optimizer', 'SGD', str)

    # for warmup
    parser.add('warmup_epoch', 0, float)

    # for data argumentation
    parser.add('train_argument', 'resize-random_crop-hflip-color_jitter-to_tensor-normalize', str)
    parser.add('valid_argument', 'resize-to_tensor-normalize', str)
    
    # for transfer learning
    parser.add('pretrained_model_path', '', str)

    # for debugging
    parser.add('debug', False, bool)

    args = parser.get_args()

    
    ###################################################################################
    # 2. Make directories and pathes.
    ###################################################################################
    log_dir = './experiments/logs/'
    model_dir = f'./experiments/models/{args.tag}/'

    log_path = log_dir + f'{args.tag}.txt'      

    if args.reset and os.path.isdir(log_dir):
        os.remove(log_path)
        shutil.rmtree(model_dir)
    
    log_dir = io_utils.create_directory(log_dir)
    model_dir = io_utils.create_directory(model_dir)

    ###################################################################################
    # 3. Set the seed number and define log function. 
    ###################################################################################
    torch_utils.set_seed(args.seed)
    log_func = lambda string='': log_utils.log_print(string, log_path)

    