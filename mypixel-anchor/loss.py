import torch
import torch.nn as nn
import math

#---------------------------pixel_loss-------------------
def get_dice_loss(gt_score, pred_score):
    # print('--------------------------------进入get_dice_loss中的torch.sum(pred_score):',torch.sum(pred_score))
    inter = torch.sum(gt_score * pred_score)
    # print('--------------------------pixel_loss中分类损失中get_dice_loss的inter:',inter)

    union = torch.sum(gt_score) + torch.sum(pred_score) + 1e-5
    # print('--------------------------pixel_loss中分类损失中get_dice_loss的union:', union)

    return 1. - (2 * inter / union)

def get_geo_loss(gt_geo, pred_geo):
    d1_gt, d2_gt, d3_gt, d4_gt, angle_gt = torch.split(gt_geo, 1, 1)
    d1_pred, d2_pred, d3_pred, d4_pred, angle_pred = torch.split(pred_geo, 1, 1)
    area_gt = (d1_gt + d2_gt) * (d3_gt + d4_gt)
    # print('--------------------------------get_geo_loss中area_gt:',angle_gt)
    area_pred = (d1_pred + d2_pred) * (d3_pred + d4_pred)

    # print('---------------------------------get_pred_loss中的area_prea:',angle_pred)
    w_union = torch.min(d3_gt, d3_pred) + torch.min(d4_gt, d4_pred)
    h_union = torch.min(d1_gt, d1_pred) + torch.min(d2_gt, d2_pred)
    area_intersect = w_union * h_union
    # print('---------------------------------get_geo_loss中的area_intersect:',area_intersect)


    area_union = area_gt + area_pred - area_intersect

    # print('----------------------------------get_geo_loss中area_union:',area_union)


    iou_loss_map = -torch.log((area_intersect + 1.0) / (area_union + 1.0))
    # print('-----------------------------get_geo_loss中iou_loss_map:',iou_loss_map)
    angle_loss_map = 1 - torch.cos(angle_pred - angle_gt)
    # print('-----------------------------get_geo_loss中angle_loss_map:',angle_loss_map)

    return iou_loss_map, angle_loss_map

class PixelLoss(nn.Module):
    def __init__(self, weight_angle=10):
        super(PixelLoss, self).__init__()
        self.weight_angle = weight_angle

    def forward(self, gt_score, pred_score, gt_geo, pred_geo, ignored_map,attention_gt,pred_attention):
        # print('----------------------------------进入pixelloss--------------------')
        print('-----------------------------------pixel_loss中的torch.sum（gt_score）----------------:',torch.sum(gt_score))
        # print('-----------------------------------pixel_loss中由网络输出作为输入传入的pred_score:',pred_score)
        print('-----------------------------------------------------torch.sum(pred_score):',torch.sum(pred_score))
        if torch.sum(gt_score) < 1:
            return torch.sum(pred_score + pred_geo) * 0
        # if math.isnan(torch.sum(pred_score).cpu()):
        #     return 0
        # print('---------------------------pixel_loss中的torch.sum(gt_score)--------:',torch.sum(gt_score))
        classify_loss = get_dice_loss(gt_score, pred_score * (1 - ignored_map))+get_dice_loss(attention_gt,pred_attention*(1-ignored_map))
        iou_loss_map, angle_loss_map = get_geo_loss(gt_geo, pred_geo)

        # print('----------------------pixel_loss中经过get_geo_loss后的iou_loss_map:',iou_loss_map)
        #
        #
        # print('----------------------pixel_loss中经过get_geo_loss后的angle_loss_map:',angle_loss_map)


        angle_loss = torch.sum(angle_loss_map * gt_score) / torch.sum(gt_score)
        iou_loss = torch.sum(iou_loss_map * gt_score) / torch.sum(gt_score)

        # angle_loss = torch.sum(angle_loss_map )
        # iou_loss = torch.sum(iou_loss_map )


        geo_loss = self.weight_angle * angle_loss + iou_loss

        print('classify_loss:',classify_loss)
        print('angel_loss:',angle_loss)
        print('iou_loss:',iou_loss)
        print('classify loss is {:.8f}, angle loss is {:.8f}, iou loss is {:.8f}'.format(classify_loss, angle_loss,
                                                                                   iou_loss))
        return geo_loss + classify_loss
#------------------------anchor_loss---------------
# from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.anchorutils import one_hot_embedding, one_hot_v3
from torch.autograd import Variable



device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
class FocalLoss(nn.Module):
    def __init__(self):
        print('-----------------------------anchor_loss-------------------------')
        super(FocalLoss, self).__init__()
        self.num_classes = 1
    def focal_loss(self, x, y):
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2
        t = one_hot_embedding(y.data.cpu(), self.num_classes + 1)  # [N,21]

        t = t[:, 1:]  # exclude background
        t=t.to(device)
        # t = Variable(t).cuda()# [N,20]
        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1 - pt).pow(gamma)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False)

    def focal_loss_alt(self, x, y):
        '''Focal loss alternative.

        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].

        Return:
          (tensor) focal loss.
        '''

        alpha = 0.25
        gamma = 2
        t = one_hot_embedding(y.data.cpu(), self.num_classes + 1)
        # print('-------------------------转换为one_hot后标签对应size():',t.shape)
        t = t[:, 1:]
        # print('--------------------------转换为one_hou后t[:,1:].shape:',t.shape)
        #-------------------------转换为one_hot后标签对应size(): torch.Size([1312227, 2])
        #--------------------------转换为one_hou后t[:,1:].shape: torch.Size([1312227, 1])
        # print('----------------------------focal_loss中传入的x.size():',x.size())

        t=t.to(device)


        xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
        pt = (2 * xt + 1).sigmoid()

        #----------------------修改版本----------------
        # p = x.sigmoid()
        # pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        # w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        # loss =- w * (1 - pt).pow(gamma)*torch.log(pt)

        #----------------------修改版本---------



        # print('---------------------------------focal_loss中focal_loss_alt的pt:',pt,'pt.sum()',pt.sum(),'pt.log().sum():',pt.log().sum())
        #
        w = alpha * t + (1 - alpha) * (1 - t)
        # print('---------------------------------focal_loss中focal_loss_alt的w:', w,'w.sum():',w.sum(),'w.size():',w.size())
        #
        # print('----------------------------------pt.log().sum():',pt.log().sum(),'pt.log.size():',pt.log().size())
        #
        # print('---------------------------------(w * pt).sum():',(w * pt).sum())
        loss = -w * pt.log() / 2
        # print('---------------------------------focal_loss中返回的loss.size():',loss.size())
        #
        # print('---------------------------------focal_loss中返回的loss.mean():',loss.mean())
        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 8].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        # print('------------Focal_loss中的loc_preds.size():',loc_preds.size())
        # print('------------Focal_loss中的loc_targets.size():',loc_targets.size())
        # print('------------Focal_loss中的cla_preds.zie():',cls_preds.size())
        # print('------------Focla_loss中的cla_targets.size():',cls_targets.size())

        batch_size, num_boxes = cls_targets.size()
        # print('------------------------------focal_loss中传入的cls_targes:',cls_targets)
        pos = cls_targets > 0  # [N,#anchors]
        # print('-------------------------------focal_loss中的pos:',pos)
        num_pos = pos.data.float().sum()# 寻找所给batch大小中标签为1的anchor个数

        print('-------------------------------------focal_loss中的num_pos:',num_pos)

        # if num_pos==0:
        #     return 0
        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,8] 获得所给batch中所有标签为1的anchor的坐标，并扩展到包含4个顶点，8个坐标的信息，的掩码
        # print('--------------------------------------focal_loss中回归部分的掩码mask.size():',mask.size())

        masked_loc_preds = loc_preds[mask].view(-1, 8)  # [#pos,8]
        # print('---------------------------------------focal_loss中的masked_loc_preds:',masked_loc_preds)
        masked_loc_targets = loc_targets[mask].view(-1, 8)  # [#pos,8]
        # print('---------------------------------------focla_loss中的masked_loc_targets:',masked_loc_targets)

        # print('-----------------------------------masked_loc_preds.size():',masked_loc_preds.size())
        # print('-----------------------------------masked_loc_targets.size():',masked_loc_targets.size())
        loc_loss = F.smooth_l1_loss(masked_loc_preds.float(), masked_loc_targets.float(), size_average=False)
        # loc_loss *= 0.5  # TextBoxes++ has 8-loc offset

        print('------------------------------Focal_loss中的loc_loss:',loc_loss)
        ################################################################
        # cls_loss = FocalLoss(loc_preds, loc_targets)
        ################################################################
        pos_neg = cls_targets > -1  # exclude ignored anchors

        # print('--------------------pos_neg.size():',pos_neg.size())

        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)#扩展到于网络输出的标签值维度一致，以便掩码操作，取对应网络输出类别值
        # print('-----------------------focal_loss中cls部分中的cls_preds:', cls_preds.size())
        # print('-----------------------focal_loss中cls部分中的mask.size():',mask.size())
        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)

        # print('-----------------------------anchor_loss中的focal_loss_alt中masked_cls_preds.size():',masked_cls_preds.size())
        #
        # print('-----------------------------anchor_loss中的focal_loss_alt中cls_targes[pos_neg]:',cls_targets[pos_neg].size(),'cls_targets.size():',cls_targets.size(),'pos_neg.size():',pos_neg.size())
        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])
        print('------------------------------Focal_loss中的cls_loss:', cls_loss)
        #pos_neg为产生标签中取出标签值为-1后剩余的标签,去除-1后只有0和1
        #anchor_loss中的focal_loss_alt中cls_targes[pos_neg]: torch.Size([1306511]) cls_targets.size(): torch.Size([32, 41109]) pos_neg.size(): torch.Size([32, 41109])



        # cls_loss=torch.nn.functional.binary_cross_entropy(masked_cls_preds.to(device),cls_targets[pos_neg].to(device)).sum()

        # print('--------------------------------focal_loss中的cls_loss:')

        # anchor_loss=0.2*(loc_loss / num_pos)+ (cls_loss / num_pos)

        anchor_loss=0.2*loc_loss+cls_loss

        # print('---------------------------------focal_loss中的anchor_loss：')
        return anchor_loss# pixel-anchor中的要求。

def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.max()
    return torch.log(torch.sum(torch.exp(x - x_max), 1, keepdim=True)) + x_max




class OHEM_loss(nn.Module):
    def __init__(self):
        super(OHEM_loss, self).__init__()
        self.num_classes = 1
        self.negpos_ratio = 3


    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 8].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''
        print('---------------------------传入forward的loc_preds.size():',loc_preds.size())
        print('---------------------------传入forward的loc_targets.size():',loc_targets.size())


        print('----------------------------传入forward的cls_preds.size():',cls_preds.size())

        print('---------------------------传入forward的cls_targets.size()-----------:',cls_targets.size())
        cls_targets = cls_targets.clamp(0, 1)  # remove ignore (-1)
        print('---------------------------cls_targets.clamp(0,1)------------：',cls_targets)
        pos = cls_targets > 0
        print('--------------------------pos---------------------:',pos)
        num_pos = pos.sum(dim=1, keepdim=True)
        print('---------------------------num_pos----------------:',num_pos,'num_pos.size():',num_pos.size())
        print('---------------------------pos.dim()--------------:',pos.dim(),'pos.size():',pos.size())


        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,8]

        print('---------------------------pos.unsqueeze前----------------:',pos)
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_preds)
        print('---------------------------pos.unsqueeze后----------------:',pos_idx)
        print('----------------------------pos_idx.size()----------------:',pos_idx.size())

        print('-----------------------------loc_preds[pos_idx].size()----:',loc_preds[pos_idx].size())

        masked_loc_preds = loc_preds[pos_idx].view(-1, 8)
        print('-----------------------masked_loc_preds:',masked_loc_preds)
        masked_loc_targets = loc_targets[pos_idx].view(-1, 8)
        print('-----------------------masked_loc_targets:', masked_loc_targets)
        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets,reduction='sum')# 选取置信度大于0的做smooth_l1_loss
        print('---------------------loc_loss-------------------:',loc_loss)

        # Compute max conf across batch for hard negative mining
        num = loc_preds.size(0)# batch数
        print('--------------------------num.size()---------------:',num)

        print('------------------------------view之前的cls_preds:',cls_preds.size())
        batch_conf = cls_preds.view(-1, self.num_classes)
        print('--------------------------batch_conf().size()--------------:',batch_conf.size())

        print('--------------------------OHEM中的cls_targets.size():',cls_targets.size())

        print('--------------------------cls_targets.view(-1,1)------:',cls_targets.view(-1,1),'cls_targets.view(-1,1).size():',cls_targets.view(-1,1).size())

        print('--------------------------log_sum_exp(batch_conf):',log_sum_exp(batch_conf))
        print('---------------------------batch_conf.gather(1, cls_targets.view(-1, 1):',batch_conf.gather(1, cls_targets.view(-1, 1)))
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, cls_targets.view(-1, 1))

        # Hard Negative Mining
        loss_c = loss_c.view(num, -1)
        loss_c[pos] = 0  # filter out pos boxes for now


        #trick积累  连续两次用torch.sort
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)

        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(cls_preds)
        neg_idx = neg.unsqueeze(2).expand_as(cls_preds)


        conf_p = cls_preds[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = cls_targets[(pos + neg).gt(0)]
        print('----------------------conf_p.size()------------------------',conf_p.size())
        print('----------------------targets_weighted()-------------------',targets_weighted.size())
        cls_loss = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = num_pos.float().sum()
        loc_loss /= N
        cls_loss /= N
        total_loss=0.2*loc_loss+cls_loss
        return total_loss



class Pixel_anchor_loss(nn.Module):
    def __init__(self,wight=3):
        super(Pixel_anchor_loss,self).__init__()
        self.Pixel_loss=PixelLoss()
        self.Anchor_loss=FocalLoss()
        self.wight=wight
    def forward(self,gt_score, pred_score, gt_geo, pred_geo, ignored_map,loc_preds, loc_targets, cls_preds, cls_targets):
        pixel_loss=self.Pixel_loss(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
        anchor_loss=self.Anchor_loss(loc_preds, loc_targets, cls_preds, cls_targets)

        total_loss=pixel_loss*self.wight+anchor_loss
        return total_loss




#------------------------------------original_loss---------------------
class OriginalFocalLoss(nn.Module):
    def __init__(self):
        super(OriginalFocalLoss, self).__init__()
        self.num_classes = 1

    def focal_loss(self, x, y):
        print('------------------输入x.size():',x.size())
        print('------------------输入y.size():',y.size())
        '''Focal loss.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        gamma = 2

        t = one_hot_embedding(y.data.cpu(), self.num_classes + 1)  # [N,21]
        # print('-----------one_hot_embedding---t.size():',t.size())

        t = t[:, 1:]  # exclude background
        # print('-----------one_hot_embedding---t.size():',t.size())
        t = t.to(device)  # [N,20]
        p = x.sigmoid()
        # print('------------x.size():',x.size())
        # print('------------t.size():',t.size())
        pt = p * t + (1 - p) * (1 - t)  # pt = p if t > 0 else 1-p
        w = alpha * t + (1 - alpha) * (1 - t)  # w = alpha if t > 0 else 1-alpha
        w = w * (1 - pt).pow(gamma)
        loss = - w*torch.log(pt)
        return loss.sum()

    def focal_loss_alt(self, x, y):
        print('----------------focal_loss_alt中的输入x.size():',x.size())
        print('----------------focal_loss_alt中的输入y.size():',y.size())
        '''Focal loss alternative.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) focal loss.
        '''
        alpha = 0.25
        t = one_hot_embedding(y.data.cpu(), self.num_classes + 1)
        print('-----------one_hot_embedding---t.size():', t.size())

        t = t[:, 1:]
        print('-----------t[:,1:]---t.size():', t.size())


        #debug()


        t = t.to(device)

        xt = x * (2 * t - 1)  # xt = x if t > 0 else -x
        pt = (2 * xt + 1).sigmoid()

        w = alpha * t + (1 - alpha) * (1 - t)
        loss = -w * pt.log() / 2
        return loss.sum()

    def forward(self, loc_preds, loc_targets, cls_preds, cls_targets):
        '''Compute loss between (loc_preds, loc_targets) and (cls_preds, cls_targets).
        Args:
          loc_preds: (tensor) predicted locations, sized [batch_size, #anchors, 8].
          loc_targets: (tensor) encoded target locations, sized [batch_size, #anchors, 8].
          cls_preds: (tensor) predicted class confidences, sized [batch_size, #anchors, #classes].
          cls_targets: (tensor) encoded target labels, sized [batch_size, #anchors].
        loss:
          (tensor) loss = SmoothL1Loss(loc_preds, loc_targets) + FocalLoss(cls_preds, cls_targets).
        '''

        batch_size, num_boxes = cls_targets.size()
        pos = cls_targets > 0  # [N,#anchors]
        num_pos = pos.data.float().sum()
        print('-------------------------------OriginalFocalloss中的num_pos:',num_pos)

        ################################################################
        # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
        ################################################################
        mask = pos.unsqueeze(2).expand_as(loc_preds)  # [N,#anchors,8]
        masked_loc_preds = loc_preds[mask].view(-1, 8)  # [#pos,8]
        masked_loc_targets = loc_targets[mask].view(-1, 8)  # [#pos,8]

        loc_loss = F.smooth_l1_loss(masked_loc_preds, masked_loc_targets, size_average=False)
        # TextBoxes++ has 8-loc offset
        ################################################################
        # cls_loss = self.focal_loss(cls_preds, cls_targets)
        ################################################################
        pos_neg = cls_targets > -1 # exclude ignored anchors
        print('----------------pos_neg.size()：',pos_neg.size())
        mask = pos_neg.unsqueeze(2).expand_as(cls_preds)
        print('-----------------mask.size():',mask.size())

        masked_cls_preds = cls_preds[mask].view(-1, self.num_classes)

        print('-------------------masked_cls_preds.size():',masked_cls_preds.size())
        print('-------------------cls_targets[pos_neg].size():',cls_targets[pos_neg].size())

        cls_loss = self.focal_loss_alt(masked_cls_preds, cls_targets[pos_neg])

        #-----------------------用BCElosss---------------
        # cls_loss=torch.nn.functional.binary_cross_entropy(masked_cls_preds,cls_targets[pos_neg])
        #-----------------------用BCEloss
        print('-------------------------------Focal_loss中的loc_loss:',loc_loss)
        print('-------------------------------Focal_loss中的cls_loss:',cls_loss)
        anchor_loss = 0.2*(loc_loss / num_pos) + (cls_loss / num_pos)
        print('-------------------------------修改pixel-anchor权重超参后的anchor_loss:',anchor_loss)
        return anchor_loss






def Debug():
    loc_preds = torch.randn((2, 2, 8))
    print('----------------------loc_preds----------------------:',loc_preds)
    loc_targets = torch.randn((2, 2, 8))
    print('----------------------loc_targets--------------------:',loc_targets)

    cls_preds = torch.randn((2, 2,2))
    print('----------------------cls_preds----------------------:',cls_preds)

    cls_targets = torch.randint(0, 2, (2, 2)).type(torch.LongTensor)
    print('---------------------debug中的-cls_targets---------------------:',cls_targets)

    print('----------------------debug中的cls_targets.data------------------------:',cls_targets.data)
    ohem = OHEM_loss()
    ohem.forward(loc_preds, loc_targets, cls_preds, cls_targets)
# Debug()
