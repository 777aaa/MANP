import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.utils.weight_norm import WeightNorm
class ClassifierCombo(nn.Module):
    def __init__(self, in_dim, n_classes, c_type, temp=10.0):
        super().__init__()
        if c_type == 'cosine':
            self.classifier = nn.Linear(in_dim, n_classes, bias = False)
            WeightNorm.apply(self.classifier, 'weight', dim=0) #split the weight update component to direction and norm
        elif c_type == 'linear':
            self.classifier = nn.Linear(in_dim, n_classes, bias = True)
        elif c_type == 'mlp':
            self.classifier = [nn.Linear(in_dim, 1024),nn.Tanh(),nn.Linear(1024, n_classes)]
            self.classifier = nn.Sequential(*self.classifier)
        # https://github.com/wyharveychen/CloserLookFewShot/blob/e03aca8a2d01c9b5861a5a816cd5d3fdfc47cd45/backbone.py#L22
        # https://github.com/arjish/PreTrainedFullLibrary_FewShot/blob/main/classifier_full_library.py#L44

        self.c_type = c_type
        self.temp = nn.Parameter(torch.tensor(temp),requires_grad=False)

    def forward(self, feat):
        if self.c_type in ['linear','mlp']:
            return self.classifier(feat)
        else:
            return self.temp * self.classifier(F.normalize(feat,dim=-1))
class SemAlign(nn.Module):
    def __init__(self, v_size, s_size, h_size=2048, drop=0.1,n_class=64):
        super(SemAlign, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(v_size + s_size, h_size),#v-v ；s-s;v'+s'-h
            nn.LeakyReLU(0.2))
        self.drop = nn.Dropout(drop)
        self.fc = nn.Linear(h_size, v_size)
        self.cls_classifier = ClassifierCombo(640,n_class, 'linear')#这里每个数据集不一样的，tiered是351

    def forward(self, semantic, contexts):
        input = torch.cat((semantic, contexts), -1)
        fusion = self.model(input)
        fusion = self.drop(fusion)
        fusion = self.fc(fusion)
        cls_logit = self.cls_classifier(fusion)
        return fusion,cls_logit
        #return fusion
