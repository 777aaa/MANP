from __future__ import print_function

import os
import numpy as np
import argparse
import socket
import time
import sys
from tqdm import tqdm
import pdb

import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from architectures.NetworkPre import FeatureNet
from trainer.FSEval import run_test_fsl
from util import adjust_learning_rate, accuracy, AverageMeter
from sklearn import metrics


class MetaTrainer(object):
    def __init__(self, args, dataset_trainer, eval_loader=None, hard_path=None):
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpus)
        args.logroot = os.path.join(args.logroot, args.featype + '_' + args.dataset)
        if not os.path.isdir(args.logroot):
            os.makedirs(args.logroot)

        try:
            iterations = args.lr_decay_epochs.split(',')
            args.lr_decay_epochs = list([])
            for it in iterations:
                args.lr_decay_epochs.append(int(it))
        except:
            pass
        
        args.model_name = '{}_{}_{}_shot_{}_k_{}_{}'.format(args.dataset, args.n_train_runs,args.n_test_runs, args.n_shots,args.k,args.epochs)

        
        self.save_path = os.path.join(args.logroot, args.model_name)
        if not os.path.isdir(self.save_path):
            os.mkdir(self.save_path)
        
        assert args.pretrained_model_path is not None, 'Missing Pretrained Model'
        full_params = torch.load(args.pretrained_model_path)
        params = full_params['feature_params']
        fusion_params = full_params['fusion_params']
        #feat_params = {k: v for k, v in params.items() if 'feature' in k}
        #cls_params = {k: v for k, v in params.items() if 'cls_classifier' in k}
        feat_params = {k.replace('module.','', 1): v for k, v in params.items() if 'feature' in k}
        cls_params = full_params#{k.replace('module.','', 1): v for k, v in params.items() if 'cls_classifier' in k}

        self.args = args
        self.train_loader, self.val_loader, n_cls = dataset_trainer
        self.model = FeatureNet(args, args.restype, n_cls, (cls_params,self.train_loader.dataset.vector_array),fusion_params)
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('number of params:', n_parameters)


        ##### Load Pretrained Weights for Feature Extractor
        model_dict = self.model.state_dict()
        model_dict.update(feat_params)
        self.model.load_state_dict(model_dict)

        self.model.train()
        print('Loaded Pretrained Weight from %s' % args.pretrained_model_path)          

        # optimizer
        if self.args.tunefeat == 0.0:
            optim_param = [{'params': self.model.cls_classifier.parameters()}]
        else:
            optim_param = [{'params': self.model.cls_classifier.parameters()},{'params': filter(lambda p: p.requires_grad, self.model.feature.parameters()),'lr': self.args.tunefeat}
                           ,{'params': filter(lambda p: p.requires_grad, self.model.fusion.parameters()),'lr': self.args.tunefeat}]

        self.optimizer = optim.SGD(optim_param, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)
        
        if torch.cuda.is_available():
            if args.n_gpu > 1:
                self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()
            cudnn.benchmark = True

        # set cosine annealing scheduler
        if args.cosine:
            print("==> training with plateau scheduler ...")
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'max')
        else:
            print("==> training with MultiStep scheduler ... gamma {} step {}".format(args.lr_decay_rate, args.lr_decay_epochs))
        #fusion
        if 'ImageNet' in args.dataset:
            #/root/data/jbw/SemFew/semantic/root/wjg/jbw/SEMOP/SemFew/semantic
            self.semantic = torch.load('/root/wjg/jbw/SEMOP/SemFew/semantic/imagenet_semantic_clip_gpt.pth')['semantic_feature']
        else:
            self.semantic = torch.load('/root/wjg/jbw/SEMOP/SemFew/semantic/cifar100_semantic_clip_gpt.pth')['semantic_feature']
        self.semantic = {k: v.float() for k, v in self.semantic.items()}
    def train(self, eval_loader=None):

        trlog = {}
        trlog['args'] = vars(self.args)
        trlog['maxmeta_acc'] = 0.0
        trlog['maxmeta_acc_epoch'] = 0
        trlog['maxmeta_auroc'] = 0.0
        trlog['maxmeta_auroc_epoch'] = 0
        trlog['maxmeta_acc_auroc'] = 0.0
        trlog['maxmeta_acc_auroc_epoch'] = 0
        
        writer = SummaryWriter(self.save_path)
        # 创建日志文件
        log_file_path = os.path.join(self.save_path, 'train_msg.txt')
        with open(log_file_path, 'w') as log_file:
            log_file.write("Training Log\n")
            log_file.write("=" * 50 + "\n")

        criterion = nn.CrossEntropyLoss()
        criterion = criterion.cuda()


        for epoch in range(1, self.args.epochs + 1):
            if self.args.cosine:
                self.scheduler.step(trlog['maxmeta_acc'])
            else:
                adjust_learning_rate(epoch, self.args, self.optimizer, 0.0001)
            
            train_acc, train_auroc, train_loss, train_msg = self.train_episode(epoch, self.train_loader, self.model, criterion, self.optimizer, self.args,self.semantic)
            writer.add_scalar('train/acc', float(train_acc), epoch)
            writer.add_scalar('train/auroc', float(train_auroc), epoch)
            writer.add_scalar('train/loss_cls', float(train_loss[0]), epoch)
            writer.add_scalar('train/loss_funit', float(train_loss[1]), epoch)
            writer.add_scalar('train/loss', float(train_loss[2]), epoch)
            writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            self.model.eval()

            #evaluate
            if eval_loader is not None:
                start = time.time()
                assert self.args.featype == 'OpenMeta'
                config = {'auroc_type':['prob']}
                result = run_test_fsl(self.model, eval_loader, config,self.semantic)
                meta_test_acc = result['data']['acc']
                open_score_auroc = result['data']['auroc_prob']
                test_acc_auroc = 0.5 * meta_test_acc[0] +0.5 * open_score_auroc[0]
                test_time = time.time() - start
                writer.add_scalar('meta/close_acc', float(meta_test_acc[0]), epoch)
                writer.add_scalar('meta/close_std', float(meta_test_acc[1]), epoch)
                writer.add_scalar('meta/open_auroc', float(open_score_auroc[0]), epoch)
                writer.add_scalar('meta/open_std', float(open_score_auroc[1]), epoch)
                writer.add_scalar('meta/acc_auroc', float(test_acc_auroc), epoch)
                
                meta_msg = 'Meta Test Acc: {:.4f}, Test std: {:.4f}, AUROC: {:.4f},ACC_AUROC:{:.4f}, Time: {:.1f}'.format(meta_test_acc[0], meta_test_acc[1], open_score_auroc[0],test_acc_auroc, test_time)
                train_msg = train_msg + ' | ' + meta_msg

                if trlog['maxmeta_acc_auroc'] < test_acc_auroc:
                    trlog['maxmeta_acc_auroc'] = test_acc_auroc
                    trlog['maxmeta_acc_auroc_epoch'] = epoch
                    acc_auroc = (meta_test_acc[0], open_score_auroc[0])
                    self.save_model(epoch, 'max_acc_auroc', acc_auroc)

                if trlog['maxmeta_acc'] < meta_test_acc[0]:
                    trlog['maxmeta_acc'] = meta_test_acc[0]
                    trlog['maxmeta_acc_epoch'] = epoch
                    acc_auroc = (meta_test_acc[0], open_score_auroc[0])
                    self.save_model(epoch, 'max_acc', acc_auroc)
                if trlog['maxmeta_auroc'] < open_score_auroc[0]:
                    trlog['maxmeta_auroc'] = open_score_auroc[0]
                    trlog['maxmeta_auroc_epoch'] = epoch
                    acc_auroc = (meta_test_acc[0], open_score_auroc[0])
                    self.save_model(epoch, 'max_auroc', acc_auroc)
                
            print(train_msg)
             # 将日志写入文件
            with open(log_file_path, 'a') as log_file:
                log_file.write(f'Epoch {epoch}: {train_msg}\n')
            # regular saving
            if epoch % 1 == 0:
                #self.save_model(epoch, str(epoch) + 'checkpoint')
                best_msg = 'The Best Meta Acc {:.4f} in Epoch {}, Best Meta AUROC {:.4f} in Epoch {}, Best Meta ACC_AUROC {:.4f} in Epoch {}'.format(
                    trlog['maxmeta_acc'], trlog['maxmeta_acc_epoch'], trlog['maxmeta_auroc'], trlog['maxmeta_auroc_epoch'],
                    trlog['maxmeta_acc_auroc'], trlog['maxmeta_acc_auroc_epoch']
                )
                print(best_msg)
                # 将最佳性能信息写入日志文件
                with open(log_file_path, 'a') as log_file:
                    log_file.write(best_msg + '\n')
                

    def train_episode(self, epoch, train_loader, model, criterion, optimizer, args,semantic):
        """One epoch training"""
        model.train()
        if self.args.tunefeat==0:
            model.feature.eval()


        batch_time = AverageMeter()
        losses_cls = AverageMeter()
        losses_funit = AverageMeter()
        losses = AverageMeter() 
        acc = AverageMeter()
        auroc = AverageMeter()
        end = time.time()

        with tqdm(train_loader, total=len(train_loader), leave=False) as pbar:
            for idx, data in enumerate(pbar):
                support_data, support_label, query_data, query_label, suppopen_data, suppopen_label, openset_data, openset_label, supp_idx, open_idx, base_ids,idx_to_class,min_label = data
                #semantic 
                #train在数据处理前后都为0-63，无需加上min_label
                #print('1',idx_to_class)
                '''{52: ['n04509417'], 59: ['n07697537'], 41: ['n03924679'], 16: ['n02113712'], 8: ['n02089867'], 13: ['n02108551'], 39: ['n03888605'], 50: ['n04435653'], 7: ['n02074367'], 26: ['n03017168'], 24: ['n02823428'], 31: ['n03337140'], 21: ['n02687172'], 12: ['n02108089'], 63: ['n13133613'], 2: ['n01704323'], 18: ['n02165456'], 0: ['n01532829'], 14: ['n02108915'], 6: ['n01910747'], 30: ['n03220513'], 27: ['n03047690'], 19: ['n02457408'], 5: ['n01843383'], 22: ['n02747177'], 3: ['n01749939'], 62: ['n13054560'], 29: ['n03207743'], 11: ['n02105505'], 10: ['n02101006'], 61: ['n09246464'], 45: ['n04251144'], 49: ['n04389033'], 32: ['n03347037'], 43: ['n04067472'], 1: ['n01558993'], 58: ['n07584110'], 28: ['n03062245'], 53: ['n04515003'], 42: ['n03998194'], 46: ['n04258138'], 57: ['n06794110'], 25: ['n02966193'], 23: ['n02795169'], 34: ['n03476684'], 33: ['n03400231'], 4: ['n01770081'], 17: ['n02120079'], 35: ['n03527444'], 60: ['n07747607'], 51: ['n04443257'], 47: ['n04275548'], 54: ['n04596742'], 20: ['n02606052'], 56: ['n04612504'], 55: ['n04604644'], 44: ['n04243546'], 36: ['n03676483'], 38: ['n03854065'], 15: ['n02111277'], 48: ['n04296562'], 40: ['n03908618'], 9: ['n02091831'], 37: ['n03838899']}'''
                #if args.dataset =='miniImageNet':
                idx_to_class = {k: v[0] for k, v in idx_to_class.items()}
                #print('train_idx_to_class',idx_to_class)
                #5shot
                #print('train-support_label',support_label)
                '''tensor([[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4],
                           [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4]
                           ])'''
                #print('train-supp_idx',supp_idx)
                '''tensor([[33, 10, 23, 40, 43],
                           [22, 42, 54, 29, 34]])'''
                supp_idx_temp =supp_idx
                open_idx_temp = open_idx
                if args.dataset =='tieredImageNet':
                    supp_idx_temp = supp_idx_temp.repeat_interleave(5, dim=1)
                    open_idx_temp = open_idx_temp.repeat_interleave(5, dim=1)
                if args.n_shots ==5:
                    supp_idx_temp = supp_idx_temp.repeat_interleave(5, dim=1)
                    open_idx_temp = open_idx_temp.repeat_interleave(5, dim=1)
                    #print('supp_idx',supp_idx)
                    '''tensor([[33, 33, 33, 33, 33, 10, 10, 10, 10, 10, 23, 23, 23, 23, 23, 40, 40, 40, 40, 40, 43, 43, 43, 43, 43],
                               [22, 22, 22, 22, 22, 42, 42, 42, 42, 42, 54, 54, 54, 54, 54, 29, 29, 29, 29, 29, 34, 34, 34, 34, 34]])'''
                #print('train-supp_idx_temp',supp_idx_temp)
                suppidx_0= supp_idx_temp[0].squeeze()
                suppidx_1 = supp_idx_temp[1].squeeze()
                open_idx_0 = open_idx_temp[0].squeeze()
                open_idx_1 = open_idx_temp[1].squeeze()
                #print('suppidx_0',suppidx_0)'''tensor([33, 33, 33, 33, 33, 10, 10, 10, 10, 10, 23, 23, 23, 23, 23, 40, 40, 40, 40, 40, 43, 43, 43, 43, 43])'''
                #print('suppidx_1',suppidx_1)'''tensor([22, 22, 22, 22, 22, 42, 42, 42, 42, 42, 54, 54, 54, 54, 54, 29, 29, 29, 29, 29, 34, 34, 34, 34, 34])'''
                #print('open_idx_0',open_idx_0)'''tensor([47, 47, 47, 47, 47, 52, 52, 52, 52, 52,  6,  6,  6,  6,  6, 46, 46, 46, 46, 46,  7,  7,  7,  7,  7])'''
                #print('open_idx_1',open_idx_1)'''tensor([60, 60, 60, 60, 60, 45, 45, 45, 45, 45, 35, 35, 35, 35, 35, 44, 44, 44, 44, 44, 53, 53, 53, 53, 53])'''
                #print(idx_to_class)
                #'''{52: 'n04509417', 59: 'n07697537', 41: 'n03924679', 16: 'n02113712', 8: 'n02089867', 13: 'n02108551', 39: 'n03888605', 50: 'n04435653', 7: 'n02074367', 26: 'n03017168', 24: 'n02823428', 31: 'n03337140', 21: 'n02687172', 12: 'n02108089', 63: 'n13133613', 2: 'n01704323', 18: 'n02165456', 0: 'n01532829', 14: 'n02108915', 6: 'n01910747', 30: 'n03220513', 27: 'n03047690', 19: 'n02457408', 5: 'n01843383', 22: 'n02747177', 3: 'n01749939', 62: 'n13054560', 29: 'n03207743', 11: 'n02105505', 10: 'n02101006', 61: 'n09246464', 45: 'n04251144', 49: 'n04389033', 32: 'n03347037', 43: 'n04067472', 1: 'n01558993', 58: 'n07584110', 28: 'n03062245', 53: 'n04515003', 42: 'n03998194', 46: 'n04258138', 57: 'n06794110', 25: 'n02966193', 23: 'n02795169', 34: 'n03476684', 33: 'n03400231', 4: 'n01770081', 17: 'n02120079', 35: 'n03527444', 60: 'n07747607', 51: 'n04443257', 47: 'n04275548', 54: 'n04596742', 20: 'n02606052', 56: 'n04612504', 55: 'n04604644', 44: 'n04243546', 36: 'n03676483', 38: 'n03854065', 15: 'n02111277', 48: 'n04296562', 40: 'n03908618', 9: 'n02091831', 37: 'n03838899'}'''
                text_feature_0 = torch.stack([semantic[idx_to_class[l.item()]] for l in suppidx_0])
                text_feature_1= torch.stack([semantic[idx_to_class[l.item()]] for l in suppidx_1])
                #print(text_feature_0.shape)
                text_feature = torch.stack((text_feature_0,text_feature_1),dim=0).cuda()
                open_text_feature_0 = torch.stack([semantic[idx_to_class[l.item()]] for l in open_idx_0])
                open_text_feature_1 = torch.stack([semantic[idx_to_class[l.item()]] for l in open_idx_1])
                open_text_feature = torch.stack((open_text_feature_0,open_text_feature_1),dim=0).cuda()
                #print(text_feature.shape)#torch.Size([2,5, 512]),torch.Size([2,25, 512])
                #print(open_text_feature.shape)#torch.Size([2,5, 512]),torch.Size([2,25, 512])
                # Data Conversion & Packaging
                support_data,support_label              = support_data.float().cuda(),support_label.cuda().long()
                query_data,query_label                  = query_data.float().cuda(),query_label.cuda().long()
                suppopen_data,suppopen_label            = suppopen_data.float().cuda(),suppopen_label.cuda().long()
                openset_data,openset_label              = openset_data.float().cuda(),openset_label.cuda().long()
                supp_idx, open_idx,base_ids = supp_idx.long(), open_idx.long(),base_ids.long()
                openset_label = self.args.n_ways * torch.ones_like(openset_label)
                the_img     = (support_data, query_data, suppopen_data, openset_data)
                the_label   = (support_label,query_label,suppopen_label,openset_label)
                the_conj    = (supp_idx, open_idx)

                _, _, probs, loss = model(text_feature,open_text_feature,the_img,the_label,the_conj,base_ids,Epoch = epoch)
                query_cls_probs, openset_cls_probs = probs
                (loss_cls, loss_open_hinge, loss_funit) = loss
                loss_open = args.gamma * loss_open_hinge + args.funit * loss_funit

                loss = loss_open + loss_cls
                
                ### Closed Set Accuracy
                close_pred = np.argmax(probs[0][:,:,:self.args.n_ways].view(-1,self.args.n_ways).cpu().numpy(),-1)
                close_label = query_label.view(-1).cpu().numpy()
                acc.update(metrics.accuracy_score(close_label, close_pred),1)

                ### Open Set AUROC
                open_label_binary = np.concatenate((np.ones(close_pred.shape),np.zeros(close_pred.shape)))
                query_cls_probs = query_cls_probs.view(-1, self.args.n_ways+1)
                openset_cls_probs = openset_cls_probs.view(-1, self.args.n_ways+1)
                open_scores = torch.cat([query_cls_probs,openset_cls_probs], dim=0).cpu().numpy()[:,-1]
                auroc.update(metrics.roc_auc_score(1-open_label_binary,open_scores),1)
                
                
                losses_cls.update(loss_cls.item(), 1)
                losses_funit.update(loss_funit.item(), 1)
                losses.update(loss.item(),1)

                # ===================backward=====================
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
                # ===================meters=====================
                batch_time.update(time.time() - end)
                end = time.time()
                
                
                pbar.set_postfix({"Acc":'{0:.2f}'.format(acc.avg), 
                                "Auroc":'{0:.2f}'.format(auroc.avg), 
                                "cls_ce" :'{0:.2f}'.format(losses_cls.avg), 
                                "funit" :'{0:.4f}'.format(losses_funit.avg), 
                                "loss":'{0:.4f}'.format(losses.avg), 
                                })

        message = 'Epoch {} Train_Acc {acc.avg:.3f} Train_Auroc {auroc.avg:.3f}'.format(epoch, acc=acc, auroc=auroc)

        return acc.avg, auroc.avg, (losses_cls.avg, losses_funit.avg,losses.avg), message
    
               
    def save_model(self, epoch, name=None, acc_auroc=None):
        state = {
            'epoch': epoch,
            'cls_params': self.model.state_dict() if self.args.n_gpu==1 else self.model.module.state_dict(),
            'acc_auroc': acc_auroc
        }
        # 'optimizer': self.optimizer.state_dict()['param_groups'],
                 
        file_name = 'epoch_'+str(epoch)+'.pth' if name is None else name + '.pth'
        print('==> Saving', file_name)
        torch.save(state, os.path.join(self.save_path, file_name))
    
    
           