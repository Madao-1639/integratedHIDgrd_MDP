import argparse
import os
import time


def parse_common_args(parser):
    # Model setting
    parser.add_argument('--model_type', type=str, default='Base', help='used in model_entry.py')
    parser.add_argument('--task', type=str, default='reg', choices=['cls','reg'], help='"cls" for classification, "reg" for regression')
        # Base
    parser.add_argument('--Base_lstm_hidden_size', type=int, default=16)
    parser.add_argument('--Base_num_lstm_layers', type=int, default=1)
    parser.add_argument('--Base_lstm_dropout', type=float, default=0.0)
    parser.add_argument('--Base_dnn_hidden_size_1', type=int, default=16)
    parser.add_argument('--Base_dnn_hidden_size_2', type=int)
    parser.add_argument('--Base_activate', type=str, default='SigmoidExpBias', choices=['Sigmoid','SigmoidExpBias','SigmoidLinear','SigmoidLinearReLU','SigmoidLeakyReLU','SigmoidELU'])
    parser.add_argument('--Base_cls_thres', type=float, default=0.5)
        # SC
    parser.add_argument('--SC_dnn_hidden_size_1', type=int, default=16)
    parser.add_argument('--SC_dnn_hidden_size_2', type=int, default=16)
    
    # Data Preprocessing
    parser.add_argument('--drop_vars', nargs='*', type=int, default=[1,5,6,10,16,18,19], help='drop duplicate variables by index')
    parser.add_argument('--scaler_type', type=str, default='Standard', choices=['Standard','MinMax'],)
    parser.add_argument('--add_noise', action='store_true')
    parser.add_argument('--NoiseAfterScale', action='store_true')
    parser.add_argument('--noise_type', type=str, default='gaussian', choices=['gaussian','white gaussian'])
    parser.add_argument('--noise_param', type=int, default=0.1, help='std in gaussian, snr in white gaussian')

    # Dataset setting
    parser.add_argument('--data_type', type=str, default='Base', choices=['Base','TW','RTF','RTFTW'], help='"TW" for Time Window dataset, "RTF" for Run-To-Failure dataset')
    parser.add_argument('--drop_failure', action='store_true', help='not output failure samples (not exclusive with -F)')
    parser.add_argument('-N', type=int, default=0, help='output N+1 consecutive samples')
    parser.add_argument('-F', action='store_true', help='include failure sample in each output')
    parser.add_argument('--window_width', type=int, default=15, help='Window width for TWDataset')

    # I/O
    parser.add_argument('--log_path', type=str, default='log')
    parser.add_argument('--save_suffix', type=str, help='some comment for model')
    parser.add_argument('--load_model_fp', type=str, help='model path for pretrain or test')
    parser.add_argument('--result_dir', type=str)
    parser.add_argument('--use_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    # parser.add_argument('--verbose', action='store_true')
    return parser


def parse_train_args(parser):
    parser.add_argument('--train_fp', type=str, default='data/train_FD001.txt')
    parser.add_argument('--k_fold', type=int, default=5)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--record_HI', type=str, choices = [None,'train','val','all'])
    parser.add_argument('--record_UUTs', type=int, nargs='*')
    parser.add_argument('--record_freq', type=int, default=5)

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--num_epoch', type=int, default=200)
    parser.add_argument('--print_freq', type=int, default=10)

    # Optimizer setting
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum for sgd, alpha parameter for adam')
    parser.add_argument('--beta', default=0.999, type=float, metavar='M',
                        help='beta parameters for adam')
    parser.add_argument('--weight_decay', '--wd', default=0.0, type=float, metavar='W', help='weight decay')

    # Loss
    parser.add_argument('--cls_loss_weight', type=float, default=1.0)
    parser.add_argument('--mfe_loss_weight', type=float, default=0.05)
    parser.add_argument('--mvf_loss_weight', type=float, default=1.0)
    parser.add_argument('--mon_loss_weight', type=float, default=2.7)
    parser.add_argument('--con_loss_weight', type=float, default=4.2)
    parser.add_argument('--FocalLoss_alpha', type=float, default=0.9)
    parser.add_argument('--FocalLoss_gamma', type=float, default=2.0)
    parser.add_argument('--MVFLoss_m', type=float, default=1.0)
    parser.add_argument('--MONLoss_c', type=float, default=0.0)
    parser.add_argument('--CONLoss_c', type=float, default=0.0)
    return parser

def parse_test_args(parser):
    parser.add_argument('--test_fp', type=str, default='data/test_FD001.txt',)
    parser.add_argument('--record_HI', action='store_true')
    parser.add_argument('--record_UUTs', type=int, nargs='*')
    return parser



def get_train_args():
    parser = argparse.ArgumentParser()
    parser = parse_common_args(parser)
    parser = parse_train_args(parser)
    args = parser.parse_args()
    return args

def get_test_args():
    parser = argparse.ArgumentParser()
    parser = parse_common_args(parser)
    parser = parse_test_args(parser)
    args = parser.parse_args()
    return args



def preprocess_common_args(args):
    model_name = args.model_type
    if args.save_suffix:
        model_name = model_name + '_' + args.save_suffix
    args.model_name = model_name
    args.time_str = time.strftime(r"%Y%m%d-%H%M%S", time.localtime())
    args.input_size = 21 - len(args.drop_vars) if args.drop_vars else 21

    if not args.result_dir:
        if not os.path.exists('result'):
            os.mkdir('result')
        args.result_dir = os.path.join('result', args.model_name + '_' + args.time_str)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

def preprocess_train_args(args):
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    checkpoint_path = os.path.join('checkpoint',args.model_name)
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)
    args.checkpoint_path = checkpoint_path



def save_args(args, save_dir):
    args_path = os.path.join(save_dir, 'args.txt')
    with open(args_path, 'w') as fp:
        fp.write(str(args).replace(', ', ',\n'))



def prepare_train_args():
    args = get_train_args()
    preprocess_common_args(args)
    preprocess_train_args(args)
    save_args(args, args.checkpoint_path)
    return args

def prepare_test_args():
    args = get_test_args()
    preprocess_common_args(args)
    # preprocess_test_args(args)
    save_args(args, args.result_dir)
    return args