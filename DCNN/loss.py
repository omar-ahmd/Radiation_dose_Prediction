# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch


class Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt, clas=None, weights=None):
        possible_dose_mask = gt[1]
        pred_dose = pred[0]
        gt_dose = gt[0]
        if clas is None or weights is None:
            if (possible_dose_mask>0).sum()>0:
                pred_dose_c = pred_dose*possible_dose_mask
                gt_dose_c = gt_dose*possible_dose_mask

                L1_loss = self.L1_loss_func(pred_dose_c, gt_dose_c)

            else:
                L1_loss = self.L1_loss_func(torch.zeros(2),torch.zeros(2))
        else:
            L1_loss = 0
            for i in range(2):
                possible_dose_mask = gt[1][clas==i]
                pred_dose = pred[0][clas==i]
                gt_dose = gt[0][clas==i]
                if (possible_dose_mask>0).sum()>0:
                    pred_dose_c = pred_dose[possible_dose_mask > 0]
                    gt_dose_c = gt_dose[possible_dose_mask > 0]

                    L1_loss += weights[i]*self.L1_loss_func(pred_dose_c, gt_dose_c)

                else:
                    L1_loss += weights[i]*self.L1_loss_func(torch.zeros(2),torch.zeros(2))

        return L1_loss

class Loss_ROI(nn.Module):
    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt, clas=None, weights=None):
        if clas is None or weights is None:
            possible_dose_mask = gt[1]
            pred_dose = pred[0]
            gt_dose = gt[0]

            if (possible_dose_mask>0).sum()>0:
                pred_dose_c = pred_dose[possible_dose_mask>0]
                gt_dose_c = gt_dose[possible_dose_mask>0]

                L1_loss = self.L1_loss_func(pred_dose_c, gt_dose_c)

            else:
                L1_loss = self.L1_loss_func(torch.zeros(2),torch.zeros(2))
        else:
            L1_loss = 0
            for i in range(2):
                possible_dose_mask = gt[1][clas==i]
                pred_dose = pred[0][clas==i]
                gt_dose = gt[0][clas==i]
                if (possible_dose_mask>0).sum()>0:
                    pred_dose_c = pred_dose[possible_dose_mask > 0]
                    gt_dose_c = gt_dose[possible_dose_mask > 0]

                    L1_loss += weights[i]*self.L1_loss_func(pred_dose_c, gt_dose_c)

                else:
                    L1_loss += weights[i]*self.L1_loss_func(torch.zeros(2),torch.zeros(2))
        return L1_loss

class L1Loss(nn.Module):
    def __init__(self, sm_weight = 1., pdm_mul=False, dvh_sm_asym=False):
        super().__init__()
        self.sm_weight = sm_weight
        self.sl1 = nn.L1Loss(reduction='mean')
        self.sl2 = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt, clas=None, weights=None):

        possible_dose_mask = gt[1]
        if (possible_dose_mask>0).sum()>0:
            pred_dose = pred[0]
            gt_dose = gt[0]
            structure_masks = gt[2]

            pdmloss = self.sl1(pred_dose[possible_dose_mask>0], gt_dose[possible_dose_mask>0])    # (bs)

            sminput = pred_dose*structure_masks
            smtarget = gt_dose*structure_masks
            
            dvhloss = self.sl2(sminput[structure_masks>0], smtarget[structure_masks>0])    # (bs)
            
            return pdmloss + self.sm_weight*dvhloss
        else:
            return self.sl1(torch.zeros(1),torch.zeros(1))
