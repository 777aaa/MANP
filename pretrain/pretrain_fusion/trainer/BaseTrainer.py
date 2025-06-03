from __future__ import print_function

import os
import numpy as np
import collections
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from tensorboardX import SummaryWriter

from architectures.Network import Backbone
from architectures.LossFeat import SupConLoss
from architectures.Semfusion import SemAlign
from trainer.MetaEval import meta_evaluation
from util import adjust_learning_rate, accuracy, AverageMeter, rot_aug


def rot_aug(x):
    bs = x.size(0)
    x_90 = x.transpose(2, 3).flip(2)
    x_180 = x.flip(2).flip(3)
    x_270 = x.flip(2).transpose(2, 3)
    rot_data = torch.cat((x, x_90, x_180, x_270), 0)
    rot_label = torch.cat((torch.zeros(bs), torch.ones(bs), 2 * torch.ones(bs), 3 * torch.ones(bs)))
    return rot_data, rot_label


def model_name(args):
    model_name = '{}_{}_{}_{}_lr_{}_{}_{}'.format(args.dataset, args.lamda, args.restype, args.batch_size, args.lr,
                                                  args.mode, args.text_type)
    if args.restype in ['ViT', 'Swin']:
        model_name = '{}_Trans{}'.format(model_name, args.vit_dim)
    if args.featype == 'Contrast':
        model_name = '{}_temp_{}_even_{}'.format(model_name, args.temp, args.even)
    if args.featype == 'Entropy':
        model_name = model_name + '_bce' if args.use_bce else model_name
    return model_name


class BaseTrainer(object):
    def __init__(self, args, dataset_trainer):
        args.logroot = os.path.join(args.logroot, args.featype)
        os.makedirs(args.logroot, exist_ok=True)

        iterations = args.lr_decay_epochs.split(',')
        args.lr_decay_epochs = [int(it) for it in iterations]

        args.model_name = model_name(args)
        self.save_path = os.path.join(args.logroot, args.model_name)
        os.makedirs(self.save_path, exist_ok=True)

        self.args = args
        self.train_loader, self.val_loader, self.n_cls = dataset_trainer

        self.log_file_path = os.path.join(self.save_path, 'train_msg.txt')

        self.model = Backbone(args, args.restype, self.n_cls)
        state_dict = torch.load(args.pret_model_path)['feature_params']
        state_dict = collections.OrderedDict([(k.replace('module.', '', 1), v) for k, v in state_dict.items()])
        model_dict = self.model.state_dict()
        model_dict.update({k: v for k, v in state_dict.items() if k in model_dict})
        self.model.load_state_dict(model_dict)

        self.fusion = SemAlign(640, args.semantic_size, h_size=4096, drop=args.drop, n_class=args.n_class).cuda()
        self.criterion = {
            'feat': SupConLoss(temperature=args.temp),
            'logit': nn.BCEWithLogitsLoss() if args.use_bce else nn.CrossEntropyLoss()
        }
        sem_path = '/root/wjg/jbw/SEMOP/SemFew/semantic/{}_semantic_{}_{}.pth'.format(
            'imagenet' if 'ImageNet' in args.dataset else 'cifar100', args.mode, args.text_type)
        self.semantic = torch.load(sem_path)['semantic_feature']
        self.semantic = {k: v.float() for k, v in self.semantic.items()}

        self.optimizer = optim.Adam(self.fusion.parameters(), lr=args.lr)
        self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.step_size, gamma=0.1)

        if torch.cuda.is_available():
            if args.n_gpu > 1:
                self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
            cudnn.benchmark = True
            self.criterion = {name: loss.cuda() for name, loss in self.criterion.items()}

        self.model.eval()

    def print_log(self, msg):
        print(msg)
        with open(self.log_file_path, 'a') as log_file:
            log_file.write(msg + '\n')

    def fusion_train_epoch(self, epoch, proto_center):
        losses = AverageMeter()
        torch.cuda.empty_cache()

        with tqdm(self.train_loader, total=len(self.train_loader), leave=False) as pbar:
            for idx, (image, target, _, idx_to_class, min_label) in enumerate(pbar):
                idx_to_class = {k: v[0] for k, v in idx_to_class.items()}
                data, labels = image.cuda(), target.cuda()
                proto = torch.tensor(np.array([proto_center[idx_to_class[l.item()]] for l in labels])).cuda()
                #proto = torch.tensor(np.array([proto_center[labels]])).cuda()
                #print(labels)
                #print(idx_to_class)
                text_feature = torch.stack([self.semantic[idx_to_class[l.item()]] for l in labels]).cuda()
                with torch.no_grad():
                    img_feature, _ = self.model(data)
                self.optimizer.zero_grad()
                self.fusion.train()
                fusion_feature, cls_logit = self.fusion(text_feature, img_feature)
                recon_loss = F.l1_loss(fusion_feature, proto)
                total_loss =  recon_loss 
                total_loss.backward()
                self.optimizer.step()
                losses.update(total_loss.item(), labels.size(0))
                pbar.set_postfix({"Epoch {} Loss".format(epoch): '{0:.2f}'.format(losses.avg)})

        self.lr_scheduler.step()
        return 'Epoch {} recon_loss {:.3f}'.format(epoch, losses.avg)

    def train(self, eval_loader=None):
        with open(self.log_file_path, 'w') as f:
            f.write("Training Log\n" + "=" * 50 + "\n")
        writer = SummaryWriter(self.save_path)
        trlog = {'args': vars(self.args), 'max_1shot_meta': 0.0, 'max_5shot_meta': 0.0,
                 'max_1shot_epoch': 0, 'max_5shot_epoch': 0}
        
        data_feature = {}
        proto_center = {}
        with tqdm(self.train_loader, total=len(self.train_loader), leave=False) as pbar:
            for idx, (image, target, _, idx_to_class, min_label) in enumerate(pbar):
                idx_to_class = {k: v[0] for k, v in idx_to_class.items()}
                data, labels = image.cuda(), target.cuda()
                labels = [idx_to_class[l.item()] for l in labels]
                #print(idx_to_class)
                with torch.no_grad():
                    x, logit = self.model(data)
                for i, l in enumerate(labels):
                    data_feature.setdefault(l, []).append(x[i].detach().cpu().numpy())
        for k, v in data_feature.items():
            proto_center[k] = np.array(v).mean(0)
        self.print_log('Finished proto_feature')

        for epoch in range(1, self.args.epochs + 1):
            train_msg = self.fusion_train_epoch(epoch, proto_center)
            if eval_loader and (epoch % 1 == 0 or epoch >= 55):
                eval_1shot_loader, eval_5shot_loader = eval_loader
                best_1shot_acc = best_1shot_std = best_5shot_acc = best_5shot_std = best_1shot_k = best_5shot_k = 0
                for k in [i * 0.01 for i in range(101)]:
                    acc1, std1 = meta_evaluation(self.model, eval_1shot_loader, self.fusion, self.semantic, k)
                    acc5, std5 = meta_evaluation(self.model, eval_5shot_loader, self.fusion, self.semantic, k, shot_5=True)
                    if acc1 > best_1shot_acc:
                        best_1shot_acc, best_1shot_std, best_1shot_k = acc1, std1, k
                    if acc5 > best_5shot_acc:
                        best_5shot_acc, best_5shot_std, best_5shot_k = acc5, std5, k
                    #self.print_log(f'Epoch {epoch}: 1-shot accuracy: {acc1:.4f} (std: {std1:.4f}) at k = {k} ; 5-shot accuracy: {acc5:.4f} (std: {std5:.4f}) at k = {k}')

                writer.add_scalar('MetaAcc/1shot', best_1shot_acc, epoch)
                writer.add_scalar('MetaStd/1shot', best_1shot_std, epoch)
                writer.add_scalar('MetaAcc/5shot', best_5shot_acc, epoch)
                writer.add_scalar('MetaStd/5shot', best_5shot_std, epoch)
                meta_msg = f'Meta best Test Acc: 1-shot {best_1shot_acc:.4f} 5-shot {best_5shot_acc:.4f}, Std: {best_1shot_std:.4f} {best_5shot_std:.4f}'
                train_msg += ' | ' + meta_msg
                if trlog['max_1shot_meta'] < best_1shot_acc:
                    trlog.update({'max_1shot_meta': best_1shot_acc, 'max_1shot_epoch': epoch, 'best_1shot_k': best_1shot_k})
                    self.save_3_model(epoch, 'mini_max_meta')
                if trlog['max_5shot_meta'] < best_5shot_acc:
                    trlog.update({'max_5shot_meta': best_5shot_acc, 'max_5shot_epoch': epoch, 'best_5shot_k': best_5shot_k})
                    self.save_3_model(epoch, 'mini_max_meta_5shot')
            self.print_log(train_msg)
            if epoch % 5 == 0 or epoch == self.args.epochs:
                self.save_3_model(epoch, 'mini_last')
                self.print_log(f'The Best Meta 1(5)-shot Acc {trlog["max_1shot_meta"]:.4f}({trlog["max_5shot_meta"]:.4f}) in Epoch {trlog["max_1shot_epoch"]} k:{trlog["best_1shot_k"]}({trlog["max_5shot_epoch"]} k:{trlog["best_5shot_k"]})')
            if epoch % 15 == 0:
                self.save_3_model(epoch, '15epoch')
            if epoch % 40 == 0:
                self.save_3_model(epoch, '40epoch')
            if epoch % 50 == 0:
                self.save_3_model(epoch, '50epoch')
            torch.save(trlog, os.path.join(self.save_path, 'trlog'))

    def save_3_model(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'params': self.model.state_dict(),
            'fusion_params': self.fusion.state_dict()
        }
        file_name = '{}.pth'.format('epoch_' + str(epoch) if name is None else name)
        self.print_log(f'==> Saving {file_name}')
        torch.save(state, os.path.join(self.save_path, file_name))

        
    def train_epoch(self, epoch, train_loader, model, criterion, optimizer, args):
        """One epoch training"""
        return 0,'to be updated'
    
    def save_model(self, epoch, name=None):
        state = {
            'epoch': epoch,
            'params': self.model.state_dict()
        }     
        file_name = '{}.pth'.format('epoch_'+str(epoch) if name is None else name)
        print('==> Saving', file_name)
        torch.save(state, os.path.join(self.save_path, file_name))

    
    def eval_report(self,eval_loader,path):
        print('Loading data from', path)
        params = torch.load(path)['params']
        if 'tiered' in self.args.dataset:
            params = {'.'.join(k.split('.')[1:]):v for k,v in params.items()}
        model_dict = self.model.state_dict()
        model_dict.update(params)
        self.model.load_state_dict(model_dict)
        self.model.eval()

        eval_1shot_loader,eval_5shot_loader = eval_loader
        meta_1shot_acc, meta_1shot_std = meta_evaluation(self.model, eval_1shot_loader)
        meta_5shot_acc, meta_5shot_std = meta_evaluation(self.model, eval_5shot_loader)
        print('Linear Regression: 1(5)-shot Accuracy {:.4f}({:.4f}) Std {:.4f}({:.4f})'.format(meta_1shot_acc,meta_5shot_acc,meta_1shot_std,meta_5shot_std))
        meta_1shot_acc, meta_1shot_std = meta_evaluation(self.model, eval_1shot_loader, type_classifier='proto')
        meta_5shot_acc, meta_5shot_std = meta_evaluation(self.model, eval_5shot_loader, type_classifier='proto')
        print('Proto Classification 1(5)-shot Accuracy {:.4f}({:.4f}) Std {:.4f}({:.4f})'.format(meta_1shot_acc,meta_5shot_acc,meta_1shot_std,meta_5shot_std))
