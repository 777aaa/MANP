import torch
import torch.nn as nn
import torch.nn.functional as F
from loss.Dist import Dist

class RPLoss(nn.CrossEntropyLoss):
    def __init__(self, **options):
        super(RPLoss, self).__init__()
        self.weight_pl = float(options['weight_pl'])
        self.temp = options['temp']
        self.Dist = Dist(num_classes=351, feat_dim=640, num_centers=options['num_centers'])#这里的num——class在不同数据集是不一样的
        self.radius = 1

        self.radius = nn.Parameter(torch.Tensor(self.radius))
        #self.radius.data.fill_(0)

    def forward(self, x, y, labels=None,epoch=None):
        #labels = labels.long()
        dist = self.Dist(x,epoch=epoch)
        logits = F.softmax(dist, dim=1)
        if labels is None: return logits, 0
        loss = F.cross_entropy(dist / self.temp, labels)
        center_batch = self.Dist.centers[labels, :]
        _dis = (x - center_batch).pow(2).mean(1)
        loss_r = F.mse_loss(_dis, self.radius)
        loss = loss + self.weight_pl * loss_r

        return logits, loss
    

