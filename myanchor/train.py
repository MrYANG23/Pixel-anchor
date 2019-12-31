import torch
import time
import os
import torch
from torch.utils import data
from torch import nn
from torch.optim import lr_scheduler
from mydatasets.dataset import custom_dataset
from model.backbone import PixelAnchornet
from loss import PixelLoss,FocalLoss,OHEM_loss,Pixel_anchor_loss,NewFocalLoss,OriginalFocalLoss
import os
import time
import numpy as np
from tqdm import tqdm



def train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, interval,log_file='',pretrain=False,pretrain_pth=''):
    file_num = len(os.listdir(train_img_path))
    trainset = custom_dataset(train_img_path, train_gt_path)#可传入img的size()
    train_loader = data.DataLoader(trainset, batch_size=batch_size, \
                                   shuffle=False, num_workers=num_workers, drop_last=True)
    pixel_criterion = PixelLoss()
    # anchor_criterion=FocalLoss()
    # anchor_criterion=NewFocalLoss()
    anchor_criterion=OriginalFocalLoss()
    pixel_anchor_criterion=Pixel_anchor_loss()#

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    model = PixelAnchornet(pretrained=False)
    data_parallel = False
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    #     data_parallel = True
    model.to(device)
    if pretrain:
        print('-----------------加载断点训练模型-------------')
        checkpoint=torch.load(pretrain_pth)
        model.load_state_dict(checkpoint)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[epoch_iter // 2], gamma=0.1)

    epochs=70
    for epoch in tqdm(range(epochs,epoch_iter)):
        model.train()
        scheduler.step()
        epoch_loss = 0
        epoch_time = time.time()
        for i, (pixel_img, loc_target,cls_target) in enumerate(train_loader):
            start_time = time.time()
            pixel_img,loc_target,cls_target= pixel_img.to(device),loc_target.to(device),cls_target.to(device)

           #----------------------------------测试标签与输入输出-------------------
            print('-------------------------训练标签中的cls_target:',cls_target,'cls_target.sum():',cls_target.sum(),'cls_target.size():',cls_target.size())
            # print('-------------------------训练的到的标签cls_target.size():',cls_target.size())
            # print('-----------------------打印训练标签中非零的个数',cls_target.nonzero().size())


            # print('----------------训练中传入网络的pixel_img的大小--------------：',pixel_img.size())
            # print('-----------------训练中传入网络的anchor_input的大小--------：',anchor_input.size())
            pre_location,pre_class= model(pixel_img)
            print('--------------------------从网络中输出的pre_class.size():',pre_class.size(),'pre_class.sum():',pre_class.sum(),'pre_class.size():',pre_class.size())
            # print('-----------------------train中的torch.sum(gt_score):',torch.sum(gt_score))
            # print('--------------------------训练中从网络中出来的标签值:',pre_class,'pre_class.sum():',pre_class.sum())


            # exit()
            # pixel_loss = pixel_criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map)
            # print('gt_score.size()',gt_score.size())
            # print('pred_score.size()',pred_score.size())
            # print('gt_geo.size():',gt_geo.size())
            # print('pred_geo():',pred_geo.size())
            # print('ignored_map.size():',ignored_map.size())
            # print('---------------计算得到的pixel_loss------------:',pixel_loss,'pixel_loss',type(pixel_loss))
            #
            #
            #
            # print('------------------------计算OHEM_loss中cla_target.size():',cls_target.size())
            anchor_loss=anchor_criterion(pre_location,loc_target,pre_class,cls_target)
            #
            # # print('pre_location.size():',pre_location.size())
            # # print('loc_target.size():',loc_target.size())
            # # print('pre_class.size():',pre_class.size())
            # # print('cls_target.size():',cls_target.size())
            print('----------------计算得到的anchor_loss-----------------:',anchor_loss,'type(anchor_loss):',type(anchor_loss))
            #
            # total_loss=3*pixel_loss+anchor_loss
            # print('-----------------计算得到的total_loss------------------',total_loss,'type(total_loss):',type(total_loss))


            #------------------------------------自定义组合total_loss----------------------
            # total_loss=pixel_anchor_criterion(gt_score, pred_score, gt_geo, pred_geo, ignored_map,pre_location,loc_target,pre_class,cls_target)
            # print('-------------------------------------total_loss:',total_loss)
            epoch_loss += anchor_loss.item()
            optimizer.zero_grad()
            anchor_loss.backward()
            optimizer.step()

            print('Epoch is [{}/{}], mini-batch is [{}/{}], time consumption is {:.8f}, batch_loss is {:.8f}'.format( \
                epoch + 1, epoch_iter, i + 1, int(file_num / batch_size), time.time() - start_time, anchor_loss.item()))

        print('epoch_loss is {:.8f}, epoch_time is {:.8f}'.format(epoch_loss / int(file_num / batch_size),
                                                                  time.time() - epoch_time))
        with open(log_file,'a') as f:
            f.write('epoch is {},epoch_loss is {}\n'.format(epoch,epoch_loss / int(file_num / batch_size)))
        print(time.asctime(time.localtime(time.time())))
        print('=' * 50)
        if (epoch + 1) % interval == 0:
            state_dict = model.module.state_dict() if data_parallel else model.state_dict()
            torch.save(state_dict, os.path.join(pths_path, 'model_epoch_{}.pth'.format(epoch + 1)))

if __name__ == '__main__':
    train_img_path = os.path.abspath('/data/yanghan/ICDAR_2015/ch4_training_images')
    train_gt_path = os.path.abspath('/data/yanghan/ICDAR_2015/ch4_training_localization_transcription_gt')
    pths_path = '/tmp/myanchor/focal_flat_loss_pths'
    log_file='./focal_flat_loss_log_file'
    pretrain_pth = '/tmp/myanchor/focal_flat_loss_pths/model_epoch_70.pth'
    if not os.path.exists(pths_path):
        os.mkdir(pths_path)


    batch_size = 4
    lr = 1e-5
    num_workers = 0
    epoch_iter = 400
    save_interval = 5
    train(train_img_path, train_gt_path, pths_path, batch_size, lr, num_workers, epoch_iter, save_interval,log_file,pretrain_pth=pretrain_pth,pretrain=True)
