from __future__ import print_function

import sys, os, pdb 
import numpy as np
import scipy
from scipy.stats import t
from tqdm import tqdm

import torch
import torch.nn.functional as F

from sklearn import metrics
from sklearn.linear_model import LogisticRegression

def meta_evaluation(net, metaloader,fusion,semantic,k, type_aug='crop', type_classifier='proto',shot_5=False):
    if type_aug == 'crop':
        return meta_test(net, metaloader,fusion,semantic,k, type_classifier,shot_5)

def meta_test(net, metaloader,fusion,semantic,k, classifier='proto',shot_5=False):
    net = net.eval()
    fusion = fusion.eval()
    acc = []
    with torch.no_grad():
        with tqdm(metaloader, total=len(metaloader), leave=False) as pbar:
            for idx, data in enumerate(pbar):
                # Data Preparation
                support_data, support_label, query_data, query_label,support_id,idx_to_class,min_label  = data
                idx_to_class = {k: v[0] for k, v in idx_to_class.items()}
                supp_idx_s=support_id.squeeze()
                supp_idx_s = supp_idx_s+min_label
                #5SHOT时情况还需要考虑
                if shot_5:
                    supp_idx_s=supp_idx_s.repeat_interleave(5)
                supp_idx_s = supp_idx_s.repeat(5)
                #print('supp_idx_s',supp_idx_s)
                text_feature = torch.stack([semantic[idx_to_class[l.item()]] for l in supp_idx_s]).cuda()
                #########
                support_data = support_data.cuda()
                query_data = query_data.cuda()
                # Data Reorganization
                _, _, height, width, channel = support_data.size()
                support_data = support_data.view(-1, height, width, channel)
                query_data = query_data.view(-1, height, width, channel)
                support_label = support_label.view(-1).numpy()
                query_label = query_label.view(-1).numpy()
                
                # Feature Extracdtion
                support_features = net(support_data)[0].view(support_data.size(0), -1)
                support_fusion_features,logit = fusion(text_feature,support_features)
                query_features = net(query_data)[0].view(query_data.size(0), -1)
                support_features = k*support_features + (1-k)*support_fusion_features
                support_features = F.normalize(support_features,p=2,dim=-1).detach().cpu().numpy()
                query_features = F.normalize(query_features,p=2,dim=-1).detach().cpu().numpy()                
                if classifier.lower() in ['lr','linearregression']:
                    clf = LogisticRegression(random_state=0, solver='lbfgs', max_iter=1000, penalty='l2', multi_class='multinomial')
                    clf.fit(support_features, support_label)
                    query_pred = clf.predict(query_features)
                elif 'proto' in classifier.lower():
                    query_pred = Proto(support_features, support_label, query_features, query_label)
                else:
                    raise NotImplementedError('classifier not supported: {}'.format(classifier))

                acc.append(metrics.accuracy_score(query_label, query_pred))
                pbar.set_postfix({"Few-Shot MetaEval Acc":'{0:.2f}'.format(acc[-1])})
    return mean_confidence_interval(acc)

def Proto(support, support_ys, query, query_label):
    proto_ys = sorted(np.unique(support_ys).tolist())
    proto = []
    for cls_id in proto_ys:
        the_feat = support[support_ys==cls_id].mean(axis=0)
        proto.append(the_feat)
    proto = np.stack(proto)

    proto_norm = np.linalg.norm(proto, axis=1, keepdims=True)
    proto = proto / proto_norm
    cosine_distance = query @ proto.transpose()

    max_idx = np.argmax(cosine_distance, axis=1)
    pred = [proto_ys[idx] for idx in max_idx]
    return pred

def mean_confidence_interval(data, confidence=0.95):
    a = 100.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * t._ppf((1+confidence)/2., n-1)
    return m, h
