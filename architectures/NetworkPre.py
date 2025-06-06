import torch.nn as nn
import torch
import torch.nn.functional as F
import random
import numpy as np
import math
import pdb
from architectures.ResNetFeat import create_feature_extractor
from architectures.AttnClassifier import Classifier
from torchvision import transforms
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from architectures.Semfusion import SemAlign
import collections


class FeatureNet(nn.Module):
    def __init__(self,args,restype,n_class,param_seman,fusion_params):
        super(FeatureNet,self).__init__()
        self.args = args
        #self.base_info = None
        #self.sam_loss = args.sam_loss
        self.restype = restype
        self.n_class = n_class
        self.featype = args.featype
        self.n_ways = args.n_ways
        self.tunefeat = args.tunefeat
        self.Epoch = 0
        self.dataset = args.dataset
        self.n_shots = args.n_shots
        '''
        self.hidden_dims = [64, 160, 320, 640]
        if args.pixel_conv:
            self.conv = nn.Sequential(nn.Conv2d(self.hidden_dims[-1],
                                                self.hidden_dims[-1] // 2, kernel_size=1, stride=1, padding=0),
                                      nn.BatchNorm2d(self.hidden_dims[-1] // 2),
                                      nn.PReLU())

            for m in self.conv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
        '''
        #self.RPL_loss_temp = args.RPL_loss_temp
        #self.bpr_mix_keep_rate = args.bpr_mix_keep_rate

        #self.distance_label = torch.Tensor([i for i in range(self.n_ways)]).cuda().long()
        #self.metric = Metric_Cosine()

        self.feature = create_feature_extractor(restype,args.dataset)
        self.feat_dim = self.feature.out_dim
        #self.trplet_loss_alpha = args.trplet_loss_alpha
        self.fusion = SemAlign(640, args.semantic_size, h_size=4096, drop=args.drop,n_class=args.n_class)
        fusion_raw_model = fusion_params
        fusion_raw_model = collections.OrderedDict([ (k.replace('module.','', 1), v) for k, v in fusion_raw_model.items()])
        fusion_dict =  self.fusion.state_dict()
        fusion_state_dict = {k:v for k,v in  fusion_raw_model.items() if k in fusion_dict.keys()}
        fusion_dict.update(fusion_state_dict)
        self.fusion.load_state_dict(fusion_dict)
        print("Loaded fusion parameters")

        self.cls_classifier = Classifier(args, self.feat_dim, param_seman, args.train_weight_base) if 'OpenMeta' in self.featype else nn.Linear(self.feat_dim, n_class)

        
        assert 'OpenMeta' in self.featype
        if self.tunefeat == 0.0:
            for _,p in self.feature.named_parameters():
                p.requires_grad=False
        else:
            if args.tune_part <= 3:
                for _,p in self.feature.layer1.named_parameters():
                    p.requires_grad=False
            if args.tune_part <= 2:
                for _,p in self.feature.layer2.named_parameters():
                    p.requires_grad=False
            if args.tune_part <= 1:
                for _,p in self.feature.layer3.named_parameters():
                    p.requires_grad=False
                    

    def forward(self, text_feature,open_text_feature,the_img, labels=None, conj_ids=None, base_ids=None, test=False,Epoch =None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_feature = text_feature.to(device)
        open_text_feature = open_text_feature.to(device)
        if labels is None:
            assert the_img.dim() == 4
            return (self.feature(the_img),None)
        else:
            self.Epoch = Epoch
            return self.open_forward(text_feature,open_text_feature,the_img, labels, conj_ids, base_ids, test)
            
    
    def open_forward(self, text_feature,open_text_feature,the_input, labels, conj_ids, base_ids, test):
        # Hyper-parameter Preparation
        the_sizes = [_.size(1) for _ in the_input]
        (ne,_,nc,nh,nw) = the_input[0].size()

        # Data Preparation
        combined_data = torch.cat(the_input,dim=1).view(-1,nc,nh,nw)#([2x160, 3, 84, 84]) bs x img_num,c,h,w
        if not self.tunefeat:
            with torch.no_grad():
                combined_feat = self.feature(combined_data).detach()
        else:
            combined_feat = self.feature(combined_data)
        #print("combined_feat",combined_feat.shape)#([2x160, 640])  
        support_feat,query_feat,supopen_feat,openset_feat = torch.split(combined_feat.view(ne,-1,self.feat_dim),the_sizes,dim=1)
        #print("support_feat",support_feat.shape)#([2, 5, 640]) ([2, 25, 640])
        #print("supopen_feat",supopen_feat.shape)
        #print("text_feature",text_feature.shape)
        #print("open_text_feature",open_text_feature.shape)
        #support_feat = self.H(text_feature, support_feat)
        #print("fusion_support_feat",support_feat.shape)#torch.Size([2, 25, 640])
        (support_label,query_label,supopen_label,openset_label) = labels#0,1,2,3,4,5
        (supp_idx, open_idx) = conj_ids
        cls_label = torch.cat([query_label, openset_label], dim=1)
        test_feats = (support_feat, query_feat, openset_feat)



        ### First Task
        support_feat = support_feat.view(ne, self.n_ways, -1, self.feat_dim)
        text_feature = text_feature.view(ne ,self.n_ways,-1,512)
        #print("support_feat",support_feat.shape)#([2, 5, 1, 640]) ([2, 5, 5, 640])
        test_cosine_scores, supp_protos1, fakeclass_protos1, loss_cls, loss_funit,recip_unit = self.task_proto(text_feature,(support_feat,query_feat,openset_feat), (supp_idx,base_ids), cls_label, test=test)
        #print(test_cosine_scores[0].shape)([2, 75, 6])
        cls_protos = torch.cat([supp_protos1, fakeclass_protos1], dim=1)
        test_cls_probs = self.task_pred(test_cosine_scores[0], test_cosine_scores[1])
    

        if test:
            test_feats = (support_feat, query_feat, openset_feat)
            return test_feats, cls_protos, test_cls_probs
        

        ## Second task
        supopen_feat = supopen_feat.view(ne, self.n_ways, -1, self.feat_dim)
        open_text_feature = open_text_feature.view(ne ,self.n_ways,-1,512)
        _, supp_protos_aug, fakeclass_protos_aug, loss_cls_aug, loss_funit_aug,recip_unit = self.task_proto(open_text_feature,(supopen_feat,openset_feat,query_feat), (open_idx,base_ids), cls_label, test=test)
        
        supp_protos1 = F.normalize(supp_protos1, dim=-1)
        fakeclass_protos1 = F.normalize(fakeclass_protos1, dim=-1)
        supp_protos_aug = F.normalize(supp_protos_aug, dim=-1)
        fakeclass_protos_aug = F.normalize(fakeclass_protos_aug, dim=-1)

        ###### the SA module
        # task2's NP to find max similar in task1
        test11 = torch.bmm(fakeclass_protos_aug, supp_protos1.transpose(1,2))
        test11,_ = torch.max(test11,dim=-1)
        test11 = test11.mean() 
        test12 = torch.bmm(supp_protos1,fakeclass_protos_aug.transpose(1,2))
        test12,_ = torch.max(test12,dim=-1)
        test12 = test12.mean() 
        # task1's NP to find max in task2
        test21 = torch.bmm(fakeclass_protos1, supp_protos_aug.transpose(1,2))
        test21,_ = torch.max(test21,dim=-1)
        test21 = test21.mean() 
        test22 = torch.bmm(supp_protos_aug,fakeclass_protos1.transpose(1,2))
        test22,_ = torch.max(test22,dim=-1)
        test22 = test22.mean() 

        # SA module loss
        loss_open_hinge =  (- test11 - test12 - test21 - test22) / 2
 
        
        loss = (loss_cls+loss_cls_aug  , loss_open_hinge, loss_funit+loss_funit_aug )
        return test_feats, cls_protos, test_cls_probs, loss
    

    def task_proto(self,text_feature,features, cls_ids, cls_label,test=False,mixup_part=None):
        test_cosine_scores, supp_protos, fakeclass_protos, _, funit_distance, recip_unit = self.cls_classifier(self.fusion,text_feature,features, cls_ids, test = test ,mixup_part=mixup_part)
        (query_cls_scores,openset_cls_scores) = test_cosine_scores
        cls_scores = torch.cat([query_cls_scores,openset_cls_scores], dim=1)
        fakeunit_loss = fakeunit_compare(funit_distance,self.n_ways,cls_label)
        cls_scores,close_label,cls_label = cls_scores.view(-1, self.n_ways+1),cls_label[:,:query_cls_scores.size(1)].reshape(-1),cls_label.view(-1)
        loss_cls = F.cross_entropy(cls_scores, cls_label) 
        return test_cosine_scores, supp_protos, fakeclass_protos, loss_cls, fakeunit_loss ,recip_unit
    
    # def task_proto_no_other(self, features, cls_ids, cls_label,test=False,mixup_part=None):
    #     test_cosine_scores, supp_protos, fakeclass_protos, _, funit_distance, recip_unit  = self.cls_classifier(features, cls_ids, test = test ,mixup_part=mixup_part)
    #     (query_cls_scores,openset_cls_scores) = test_cosine_scores
    #     cls_scores = torch.cat([query_cls_scores,openset_cls_scores], dim=1)
    #     fakeunit_loss = fakeunit_compare(funit_distance,self.n_ways,cls_label)
    #     cls_scores,close_label,cls_label = cls_scores.view(-1, self.n_ways+1),cls_label[:,:query_cls_scores.size(1)].reshape(-1),cls_label.view(-1)
    #     loss_cls = F.cross_entropy(cls_scores, cls_label) 
    #     return test_cosine_scores, supp_protos, fakeclass_protos, loss_cls, fakeunit_loss,recip_unit
    
    def task_pred(self, query_cls_scores, openset_cls_scores, many_cls_scores=None):
        query_cls_probs = F.softmax(query_cls_scores.detach(), dim=-1)
        openset_cls_probs = F.softmax(openset_cls_scores.detach(), dim=-1)
        if many_cls_scores is None:
            return (query_cls_probs, openset_cls_probs)
        else:
            many_cls_probs = F.softmax(many_cls_scores.detach(), dim=-1)
            return (query_cls_probs, openset_cls_probs, many_cls_probs, query_cls_scores, openset_cls_scores)
    def get_feat_logits(self, proto, kquery, uquery, distance="pixel_sim"):
        if distance == "pixel_sim":
            kquery = kquery.view(-1, self.feat_dim[0], self.feat_dim[1] * self.feat_dim[2]
                                 ).unsqueeze(1).permute(0, 1, 3, 2).unsqueeze(-1)
            uquery = uquery.view(-1, self.feat_dim[0], self.feat_dim[1] * self.feat_dim[2]
                                 ).unsqueeze(1).permute(0, 1, 3, 2).unsqueeze(-1)
            proto = proto.squeeze().view(-1, self.feat_dim[0], self.feat_dim[1] * self.feat_dim[2]
                                         ).unsqueeze(0).unsqueeze(2)

            klogits = torch.nn.CosineSimilarity(dim=3)(kquery, proto)
            ulogits = torch.nn.CosineSimilarity(dim=3)(uquery, proto)
            if self.args.top_method == 'query':
                klogits = klogits.topk(self.args.top_k, dim=3).values.sum(dim=[2, 3]) / self.args.top_k
                ulogits = ulogits.topk(self.args.top_k, dim=3).values.sum(dim=[2, 3]) / self.args.top_k
            elif self.args.top_method == 'proto':
                klogits = klogits.topk(self.args.top_k, dim=2).values.sum(dim=[2, 3]) / self.args.top_k
                ulogits = ulogits.topk(self.args.top_k, dim=2).values.sum(dim=[2, 3]) / self.args.top_k
            else:
                klogits = klogits.topk(self.args.top_k, dim=2).values / self.args.top_k
                ulogits = ulogits.topk(self.args.top_k, dim=2).values / self.args.top_k
                klogits = klogits.topk(self.args.top_k, dim=3).values.sum(dim=[2, 3]) / self.args.top_k
                ulogits = ulogits.topk(self.args.top_k, dim=3).values.sum(dim=[2, 3]) / self.args.top_k
        else:
            raise NotImplementedError

        return klogits, ulogits



class Metric_Cosine(nn.Module):
    def __init__(self, temperature=10):
        super(Metric_Cosine, self).__init__()
        self.temp = nn.Parameter(torch.tensor(float(temperature)))

    def forward(self, supp_center, query_feature):
        ## supp_center: bs*nway*D
        ## query_feature: bs*(nway*nquery)*D
        supp_center = F.normalize(supp_center, dim=-1) # eps=1e-6 default 1e-12
        query_feature = F.normalize(query_feature, dim=-1)
        logits = torch.bmm(query_feature, supp_center.transpose(1,2))
        return logits * self.temp   



    
def fakeunit_compare(funit_distance,n_ways,cls_label):
    
    cls_label_binary = F.one_hot(cls_label)[:,:,:-1].float()
    loss = torch.sum(F.binary_cross_entropy_with_logits(input=funit_distance, target=cls_label_binary))
    return loss



