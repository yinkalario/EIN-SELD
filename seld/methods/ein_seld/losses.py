import numpy as np
import torch
from methods.utils.loss_utilities import BCEWithLogitsLoss, MSELoss


class Losses:
    def __init__(self, cfg):
        
        self.cfg = cfg
        self.beta = cfg['training']['loss_beta']

        self.losses = [BCEWithLogitsLoss(reduction='mean'), MSELoss(reduction='mean')]
        self.losses_pit = [BCEWithLogitsLoss(reduction='PIT'), MSELoss(reduction='PIT')]

        self.names = ['loss_all'] + [loss.name for loss in self.losses]
    
    def calculate(self, pred, target, epoch_it=0):

        if 'PIT' not in self.cfg['training']['PIT_type']:
            updated_target = target
            loss_sed = self.losses[0].calculate_loss(pred['sed'], updated_target['sed'])
            loss_doa = self.losses[1].calculate_loss(pred['doa'], updated_target['doa'])
        elif self.cfg['training']['PIT_type'] == 'tPIT':
            loss_sed, loss_doa, updated_target = self.tPIT(pred, target)

        loss_all = self.beta * loss_sed + (1 - self.beta) * loss_doa
        losses_dict = {
            'all': loss_all,
            'sed': loss_sed,
            'doa': loss_doa,
            'updated_target': updated_target
        }
        return losses_dict

    def tPIT(self, pred, target):
        """Frame Permutation Invariant Training for 2 possible combinations

        Args:
            pred: {
                'sed': [batch_size, T, num_tracks=2, num_classes], 
                'doa': [batch_size, T, num_tracks=2, doas=3]
            }
            target: {
                'sed': [batch_size, T, num_tracks=2, num_classes], 
                'doa': [batch_size, T, num_tracks=2, doas=3]            
            }
        Return:
            updated_target: updated target with the minimum loss frame-wisely
                {
                    'sed': [batch_size, T, num_tracks=2, num_classes], 
                    'doa': [batch_size, T, num_tracks=2, doas=3]            
                }
        """
        target_flipped = {
            'sed': target['sed'].flip(dims=[2]),
            'doa': target['doa'].flip(dims=[2])
        }

        loss_sed1 = self.losses_pit[0].calculate_loss(pred['sed'], target['sed'])
        loss_sed2 = self.losses_pit[0].calculate_loss(pred['sed'], target_flipped['sed'])
        loss_doa1 = self.losses_pit[1].calculate_loss(pred['doa'], target['doa'])
        loss_doa2 = self.losses_pit[1].calculate_loss(pred['doa'], target_flipped['doa'])

        loss1 = loss_sed1 + loss_doa1
        loss2 = loss_sed2 + loss_doa2

        loss_sed = (loss_sed1 * (loss1 <= loss2) + loss_sed2 * (loss1 > loss2)).mean()
        loss_doa = (loss_doa1 * (loss1 <= loss2) + loss_doa2 * (loss1 > loss2)).mean()
        updated_target_sed = target['sed'].clone() * (loss1[:, :, None, None] <= loss2[:, :, None, None]) + \
            target_flipped['sed'].clone() * (loss1[:, :, None, None] > loss2[:, :, None, None])
        updated_target_doa = target['doa'].clone() * (loss1[:, :, None, None] <= loss2[:, :, None, None]) + \
            target_flipped['doa'].clone() * (loss1[:, :, None, None] > loss2[:, :, None, None])
        updated_target = {
            'sed': updated_target_sed,
            'doa': updated_target_doa
        }
        return loss_sed, loss_doa, updated_target
