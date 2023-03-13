# -*- encoding: utf-8 -*-
import torch.nn as nn
import torch


class L1Loss(nn.Module):

    def __init__(self):
        super().__init__()
        self.L1_loss_func = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt, clas=None, weights=None):
        possible_dose_mask = gt[1]
        pred_dose = pred[0]
        gt_dose = gt[0]
        if clas is None or weights is None:
            if possible_dose_mask.sum()>0:
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
                if possible_dose_mask.sum()>0:
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

            if possible_dose_mask.sum()>0:
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

class Loss_SM_ROI(nn.Module):
    '''
    sum of the ROI loss and a loss of the MAE of the dose in the structure masks(SM)
    '''
    
    def __init__(self, sm_weight = 1.):
        super().__init__()
        self.sm_weight = sm_weight
        self.sl1 = nn.L1Loss(reduction='mean')
        self.sl2 = nn.L1Loss(reduction='mean')

    def forward(self, pred, gt, clas=None, weights=None):
        possible_dose_mask = gt[1]
        structure_masks = gt[2]
        if clas is None or weights is None:
    
            if possible_dose_mask.sum()>0 and structure_masks.sum()>0:
                pred_dose = pred[0]
                gt_dose = gt[0]
                

                pdmloss = self.sl1(pred_dose[possible_dose_mask>0], gt_dose[possible_dose_mask>0])    # (bs)

                sminput = pred_dose*structure_masks
                smtarget = gt_dose*structure_masks
                
                dvhloss = self.sl2(sminput[structure_masks>0], smtarget[structure_masks>0])    # (bs)
                
                return pdmloss + self.sm_weight*dvhloss
            else:
                return self.sl1(torch.zeros(1),torch.zeros(1))
        else:
            L1_loss = 0
            for i in range(2):
                possible_dose_mask = gt[1][clas==i]
                pred_dose = pred[0][clas==i]
                gt_dose = gt[0][clas==i]
                if possible_dose_mask.sum()>0:
                    pred_dose_c = pred_dose[possible_dose_mask > 0]
                    gt_dose_c = gt_dose[possible_dose_mask > 0]

                    
                    sminput = pred_dose*structure_masks

                    smtarget = gt_dose*structure_masks
                    if structure_masks.sum()>0:
                        dvhloss = self.sl2(sminput[structure_masks>0], smtarget[structure_masks>0])    # (bs)
                    else:
                        dvhloss=self.L1_loss_func(torch.zeros(2),torch.zeros(2))

                    L1_loss += weights[i]*(self.L1_loss_func(pred_dose_c, gt_dose_c)+self.sm_weight*dvhloss)

                else:
                    L1_loss += weights[i]*self.L1_loss_func(torch.zeros(2),torch.zeros(2))

#class PTV_estimator_loss(nn.Module):
#    '''
#    sum of the ROI loss and a loss of the MAE of the dose in the structure masks(SM)
#    '''
#    
#    def __init__(self, weights=None):
#        super().__init__()
#        self.loss = nn.CrossEntropyLoss(reduction='mean', weight=weights)
#        self.sl1 = nn.L1Loss()
#
#    def forward(self, pred, gt, clas=None, weights=None):
#        possible_dose_mask = gt[1]
#    
#        if possible_dose_mask.sum()>0:
#            #pred_dose = pred[0].reshape(pred[0].shape[0], 4, -1)
#            #gt_ptv = gt[0].reshape(gt[0].shape[0], 4, -1)
#
#            #possible_dose_mask = possible_dose_mask.reshape(possible_dose_mask.shape[0], 1, -1)                 
#            pdmloss = self.loss(pred[0], gt[0])    # (bs)
#            print(pred[0].shape)
#            print(gt[0].shape)
#
#            
#
#            return pdmloss
#        else:
#            return self.sl1(torch.zeros(1),torch.zeros(1))

class PTV_estimator_loss(nn.Module):
    '''
    sum of the ROI loss and a loss of the MAE of the dose in the structure masks(SM)
    '''
    
    def __init__(self, weights=None):
        super().__init__()
        self.loss = nn.CrossEntropyLoss(reduction='mean')
        self.sl1 = nn.L1Loss()

    def forward(self, pred, gt, clas=None, weights=None):
        possible_dose_mask = gt[1]
        if possible_dose_mask.sum()>0:

            pdmloss = self.loss(pred[0]*possible_dose_mask ,gt[0]*possible_dose_mask)    # (bs)
            

            return pdmloss
        else:
            return self.sl1(torch.zeros(1),torch.zeros(1))
