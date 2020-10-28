import torch
import torch.nn as nn
import torch.nn.functional as F
eps = torch.finfo(torch.float32).eps


class MSELoss:
    def __init__(self, reduction='mean'):
        self.reduction = reduction
        self.name = 'loss_MSE'
        if self.reduction != 'PIT':
            self.loss = nn.MSELoss(reduction='mean')
        else:
            self.loss = nn.MSELoss(reduction='none')
    
    def calculate_loss(self, pred, target):
        if self.reduction != 'PIT':
            return self.loss(pred, target)
        else:
            return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))


class BCEWithLogitsLoss:
    def __init__(self, reduction='mean', pos_weight=None):
        self.reduction = reduction
        self.name = 'loss_BCEWithLogits'
        if self.reduction != 'PIT':
            self.loss = nn.BCEWithLogitsLoss(reduction=self.reduction, pos_weight=pos_weight)
        else:
            self.loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)
    
    def calculate_loss(self, pred, target):
        if self.reduction != 'PIT':
            return self.loss(pred, target)
        else:
            return self.loss(pred, target).mean(dim=tuple(range(2, pred.ndim)))

