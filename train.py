import os
import torch
import argparse
import numpy as np
# from models.base_model import Base_Model

from trainer.MetaTrainer import MetaTrainer
from trainer.GMetaTrainer import GMetaTrainer
from dataloader.dataloader import get_dataloaders
from architectures.mix_test import saliencymix_test 

model_pool = ['ResNet18','ResNet12','WRN28']
parser = argparse.ArgumentParser('argument for training')

# General Setting
parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
parser.add_argument('--num_workers', type=int, default=1, help='num of workers to use')
parser.add_argument('--epochs', type=int, default=40, help='number of training epochs')
parser.add_argument('--featype', type=str, default='OpenMeta', choices=['OpenMeta', 'GOpenMeta'], help='type of task: OpenMeta -- FSOR, GOpenMeta --- GFSOR')
parser.add_argument('--restype', type=str, default='ResNet12', choices=model_pool, help='Network Structure')
parser.add_argument('--dataset', type=str, default='miniImageNet', choices=['miniImageNet', 'tieredImageNet', 'CIFAR-FS', 'FC100'])
parser.add_argument('--gpus', type=str, default='3')
parser.add_argument('-t', '--trial', type=str, default='1', help='the experiment id')

# Optimization
parser.add_argument('--cosine', action='store_true', help='using cosine annealing')
parser.add_argument('--learning_rate', type=float, default=0.07, help='learning rate')
parser.add_argument('--lr_decay_epochs', type=str, default='35', help='where to decay lr, can be a list')
parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--adam', action='store_true', help='use adam optimizer')
parser.add_argument('--tunefeat', type=float, default=0.0001, help='update feature parameter')

# Specify folder
parser.add_argument('--logroot', type=str, default='/home', help='path to save model')
parser.add_argument('--data_root', type=str, default='/home', help='path to data root')
parser.add_argument('--pretrained_model_path', type=str, default='/home', help='path to pretrained model') #mini
# Meta Setting
parser.add_argument('--n_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
parser.add_argument('--n_shots', type=int, default=1, metavar='N', help='Number of shots in test')
parser.add_argument('--n_queries', type=int, default=15, metavar='N', help='Number of query in test') ### can be set to 10 if memory out
parser.add_argument('--n_aug_support_samples', default=5, type=int, help='The number of augmented samples for each meta test sample')
parser.add_argument('--n_train_para', type=int, default=2, metavar='train_batch_size', help='Size of training batch   ')
parser.add_argument('--n_train_runs', type=int, default=300, help='Number of training episodes')
parser.add_argument('--n_test_runs', type=int, default=3000, metavar='N', help='Number of test runs')

# Meta Control
parser.add_argument('--train_weight_base', type=int, default=1, help='enable training base class weights')
parser.add_argument('--neg_gen_type', type=str, default='attg', choices=['attg', 'att', 'mlp'])
parser.add_argument('--base_seman_calib',type=int, default=1, help='base semantics calibration')
parser.add_argument('--agg', type=str, default='mlp', choices=['avg', 'mlp'])

parser.add_argument('--tune_part', type=int, default=4, choices=[1,2, 3, 4])
parser.add_argument('--base_size', default=-1, type=int)
parser.add_argument('--n_open_ways', type=int, default=5, metavar='N', help='Number of classes for doing each classification run')
parser.add_argument('--funit', type=float, default=1)
parser.add_argument('--gamma', type=float, default=0.3, help='loss cofficient for SA')
parser.add_argument('--open_weight_sum_cali', type=float, default=0.03)
parser.add_argument('--k', type=float, default=1.0)

parser.add_argument('--semantic-size', type=int, default=512)
parser.add_argument('--drop', type=float, default=0.0)
parser.add_argument('--step-size', type=int, default=40)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--mode', type=str, default='clip',
                        choices=['clip', 'bert'])
parser.add_argument('--text_type', type=str, default='gpt',
                        choices=['gpt', 'name', 'definition'])
parser.add_argument('--n_class', type=int, default=64)#teried 351

if __name__ == "__main__":
    torch.manual_seed(3407)

    args = parser.parse_args()

    args.n_train_runs = args.n_train_runs * args.n_train_para
    args.n_gpu = len(args.gpus.split(',')) 
    args.train_weight_base = args.train_weight_base==1
    args.base_seman_calib = args.base_seman_calib==1


    if args.featype == 'OpenMeta':
        open_train_val_loader, meta_test_loader, n_cls = get_dataloaders(args,'openmeta')
        dataloader_trainer = (open_train_val_loader, meta_test_loader, n_cls)
        args.base_size = n_cls if args.base_size == -1 else args.base_size
        #saliencymix_test(dataloader_trainer)
        trainer = MetaTrainer(args,dataloader_trainer,meta_test_loader)
        trainer.train(meta_test_loader)


    
