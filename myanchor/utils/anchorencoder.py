'''Encode object boxes and labels.'''
import math
import torch
import numpy as np
import time

from utils.anchorutils import meshgrid, box_iou, change_box_order, softmax
from utils.anchor_nms_poly import non_max_suppression_poly
device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")


class DataEncoder:
    def __init__(self, cls_thresh=0.1, nms_thresh=0.1):
        self.anchor_areas = [16*16., 32*32., 64*64., 128*128., 256*256, 512*512.]  # v3
        self.aspect_ratios = [1., 2., 3., 5., 7.,15.,25.,35.,1./2., 1./3., 1./5.,1./7.,1./15,1./25.,1./35.]                  # v3
        # self.anchor_areas=[16*16., 64*64., 128*128., 256*256.,256*256., 256*256.]  #myself v1版本
        # self.anchor_areas=[8*8.,32*32.,64*64.,128*128.,128*128.,128*128.] # myself v2版本


        #注意，设置的anchor大小是直接相对于原图的。

        #self.anchor_areas = [30*30., 70*70., 120*120., 250*250., 320*320. ,450*450.]  #v5
        #self.aspect_ratios = [1.0, 1.5, 2.0, 3.0, 5.0, 0.5, 0.2]                      #v5

        self.APL_list=['APL_01','APL_02','APL_03','APL_04','APL_05'] # 每个APL模块中的子模块
        #注意，不同于在最后得到的每层特征图上做标签的制作，pixel-anchor是在每个APL子模块得到的特征图上做anchor标签
        #原因在于，不同的子模块对应有不同的anchor_ratios，和不同的anchor-density
        self.FEAT_list=['feat1','feat2','feat3','feat4','feat5','feat6']# 特征图个数


        self.squre_ratios=[1]
        self.medium_vertical_ratios=[2.,3.,5.,7.]
        self.medium_horizonal_ratios=[1/2.,1/3.,1/5.,1/7.]
        self.long_vertical_ratios=[15.,25.,35.]
        self.long_horizonal_ratios=[1/15.,1/25.,1/35.]


        # self.anchor_wh = self._get_anchor_wh()#size(fms,len(aspect_ratrios),2)
        self.anchor_wh_dict=self.myself_get_anchor_wh() #dict
        self.cls_thresh = cls_thresh
        self.nms_thresh = nms_thresh
    #
    # def _get_anchor_wh(self):#返回在不同ferture上,每个cell对应的不同宽长比框的宽和长
    #     '''Compute anchor width and height for each feature map.
    #     Returns:
    #       anchor_wh: (tensor) anchor wh, sized [#fm, #anchors_per_cell, 2].
    #     '''
    #     anchor_wh = []
    #     for s in self.anchor_areas:
    #         for ar in self.aspect_ratios:  # w/h = ar
    #             anchor_h = math.sqrt(s/ar)
    #             anchor_w = ar * anchor_h
    #             anchor_wh.append([anchor_w, anchor_h])
    #     print('------------------得到所有特征图上对应不同比列的anchor宽和长的集合--------',anchor_wh)
    #     print('---------------------len(anchor_wh)----------',len(anchor_wh))
    #     num_fms = len(self.anchor_areas)
    #     return torch.FloatTensor(anchor_wh).view(num_fms, -1, 2)

    def myself_get_anchor_wh(self):
        total_anchor_wh=[]
        get_anchor_wh={}

        num_fms = len(self.anchor_areas)
        for apl_name in self.APL_list:
            if apl_name =='APL_01':
                anchor_wh1=[]
                for s in self.anchor_areas:
                    anchor_wh_perfms = []
                    for ar in self.squre_ratios:
                        anchor_h = math.sqrt(s / ar)
                        anchor_w = ar * anchor_h
                        anchor_wh_perfms.append([anchor_w, anchor_h])
                    anchor_wh1.append(anchor_wh_perfms)
                anchor_wh1=torch.FloatTensor(anchor_wh1).view(num_fms, -1, 2)#(6,1,2)
                get_anchor_wh.update({'APL_01':anchor_wh1})
            if apl_name=='APL_02':
                anchor_wh2 = []
                for s in self.anchor_areas:
                    anchor_wh_perfms = []
                    for ar in self.medium_vertical_ratios:
                        anchor_h = math.sqrt(s / ar)
                        anchor_w = ar * anchor_h

                        anchor_wh_perfms.append([anchor_w, anchor_h])
                    anchor_wh2.append(anchor_wh_perfms)
                anchor_wh2 = torch.FloatTensor(anchor_wh2).view(num_fms, -1, 2)#(6,4,2)
                get_anchor_wh.update({'APL_02': anchor_wh2})

            if apl_name=='APL_03':
                anchor_wh3 = []
                for s in self.anchor_areas:
                    anchor_wh_perfms = []
                    for ar in self.medium_horizonal_ratios:
                        anchor_h = math.sqrt(s / ar)
                        anchor_w = ar * anchor_h
                        anchor_wh_perfms.append([anchor_w, anchor_h])
                    anchor_wh3.append(anchor_wh_perfms)
                anchor_wh3 = torch.FloatTensor(anchor_wh3).view(num_fms, -1, 2)#(6,4,2)
                get_anchor_wh.update({'APL_03': anchor_wh3})
            if apl_name=='APL_04':
                anchor_wh4 = []
                for s in self.anchor_areas[1:]:
                    anchor_wh_perfms = []
                    for ar in self.long_vertical_ratios:
                        anchor_h = math.sqrt(s / ar)
                        anchor_w = ar * anchor_h
                        anchor_wh_perfms.append([anchor_w, anchor_h])
                    anchor_wh4.append(anchor_wh_perfms)

                anchor_wh4 = torch.FloatTensor(anchor_wh4).view(num_fms-1, -1, 2)#(5,3,2)
                get_anchor_wh.update({'APL_04': anchor_wh4})
            if apl_name=='APL_05':
                anchor_wh5 = []
                for s in self.anchor_areas[1:]:
                    anchor_wh_perfms = []
                    for ar in self.long_horizonal_ratios:
                        anchor_h = math.sqrt(s / ar)
                        anchor_w = ar * anchor_h
                        anchor_wh_perfms.append([anchor_w, anchor_h])
                    anchor_wh5.append(anchor_wh_perfms)
                anchor_wh5 = torch.FloatTensor(anchor_wh5).view(num_fms-1, -1, 2)#(5,3,2)
                get_anchor_wh.update({'APL_05': anchor_wh5})
        # if feat_name =='feat1':
        #     return torch.cat(total_anchor_wh[:3],dim=1)# 若是feat1 则没有long_anchor
        # print('----------------------len(get_anchor_wh)---------:',len(get_anchor_wh))
        # print(get_anchor_wh['APL_01'])
        # print(get_anchor_wh['APL_05'].size())
        return get_anchor_wh

    def myself_get_anchor_box(self,input_size,mediunm_anchor_density_list=[1.,2.,3.,4.,3.,2.],long_anchor_density_list=[4.,4.,6.,4.,3.]):


        # print('------------------------传入myself_get_anchor_box中的input_size:',input_size)# tensor([1280.,  704.])
        all_anchor_boxes=[]
        fm_sizes = [torch.Tensor(input_size/4.), torch.Tensor(input_size/16.), torch.Tensor(input_size/32.), torch.Tensor(input_size/64.),
                    torch.Tensor(input_size/64.), torch.Tensor(input_size/64.)]
        # print('---------------------------------------得到的特征图fm_sizes:',fm_sizes)

        #--------------------------------------根据输入图片的大小动态的计算每个features上的shape---------------
        for feat_name in self.FEAT_list:
            if feat_name =='feat1':
                # print('-----------------进入feat1-----------------')
                medium_density=mediunm_anchor_density_list[0]
                feat1_box=self.myself_get_APL_anchor_boxes(input_size,medium_density,long_density=None,fm_sizes=fm_sizes[0],feat_name=feat_name)
                # print('*****************************feat1_box.size():',feat1_box.size())
                all_anchor_boxes.append(feat1_box)
            if feat_name =='feat2':
                # print('----------------------进入feat2-----------------')
                medium_density = mediunm_anchor_density_list[1]
                long_density = long_anchor_density_list[0]
                feat2_box = self.myself_get_APL_anchor_boxes(input_size,medium_density,long_density,fm_sizes=fm_sizes[1],feat_name=feat_name)
                # print('*******************************feat2_box.size():',feat2_box.size())
                all_anchor_boxes.append(feat2_box)
            if feat_name =='feat3':
                # print('-----------------进入feat3-----------------')
                medium_density = mediunm_anchor_density_list[2]
                long_density = long_anchor_density_list[1]
                feat3_box = self.myself_get_APL_anchor_boxes(input_size,medium_density,long_density,fm_sizes=fm_sizes[2],feat_name=feat_name)
                # print('*******************************feat3_box.size():', feat3_box.size())
                all_anchor_boxes.append(feat3_box)
            if feat_name =='feat4':
                # print('-----------------进入feat4-----------------')
                medium_density = mediunm_anchor_density_list[3]
                long_density = long_anchor_density_list[2]
                feat4_box = self.myself_get_APL_anchor_boxes(input_size,medium_density,long_density,fm_sizes=fm_sizes[3],feat_name=feat_name)
                # print('*******************************feat4_box.size():', feat4_box.size())
                all_anchor_boxes.append(feat4_box)
            if feat_name =='feat5':
                # print('-----------------进入feat5-----------------')
                medium_density = mediunm_anchor_density_list[4]
                long_density = long_anchor_density_list[3]
                feat5_box = self.myself_get_APL_anchor_boxes(input_size,medium_density,long_density,fm_sizes=fm_sizes[4],feat_name=feat_name)
                # print('*******************************feat5_box.size():', feat5_box.size())
                all_anchor_boxes.append(feat5_box)
            if feat_name =='feat6':
                # print('-----------------进入feat6-----------------')
                medium_density = mediunm_anchor_density_list[5]
                long_density = long_anchor_density_list[4]
                feat6_box = self.myself_get_APL_anchor_boxes(input_size,medium_density,long_density,fm_sizes=fm_sizes[5],feat_name=feat_name)
                # print('*******************************feat6_box.size():', feat6_box.size())
                all_anchor_boxes.append(feat6_box)
        return torch.cat(all_anchor_boxes,dim=0)


    def myself_get_APL_anchor_boxes(self,input_size,medium_density,long_density,fm_sizes,feat_name=''):
        total_apl_boxes=[]
        for apl_name in self.APL_list:
            if apl_name =='APL_01':
                # print('---------------------进入APL_01----------------------')
                apl1_box=self.myself_get_peer_apl_anchor_box(input_size,anchor_desity=1.,aspect_ratios=self.squre_ratios,fm_sizes=fm_sizes,apl_name ='APL_01',feat_name=feat_name)
                total_apl_boxes.append(apl1_box)
            if apl_name =='APL_02':
                # print('---------------------进入APL_02----------------------')
                apl2_box=self.myself_get_peer_apl_anchor_box(input_size,anchor_desity=medium_density,aspect_ratios=self.medium_vertical_ratios,fm_sizes=fm_sizes,apl_name ='APL_02',feat_name=feat_name)
                total_apl_boxes.append(apl2_box)
            if apl_name =='APL_03':
                # print('---------------------进入APL_03----------------------')
                apl3_box=self.myself_get_peer_apl_anchor_box(input_size,anchor_desity=medium_density,aspect_ratios=self.medium_horizonal_ratios,fm_sizes=fm_sizes,apl_name ='APL_03',feat_name=feat_name)
                total_apl_boxes.append(apl3_box)
            if apl_name =='APL_04' and feat_name!='feat1':
                # print('---------------------进入APL_04----------------------')
                apl4_box=self.myself_get_peer_apl_anchor_box(input_size,anchor_desity=long_density,aspect_ratios=self.long_vertical_ratios,fm_sizes=fm_sizes,apl_name ='APL_04',feat_name=feat_name)
                total_apl_boxes.append(apl4_box)
            if apl_name =='APL_05'  and  feat_name!='feat1':
                # print('---------------------进入APL_05----------------------')
                apl5_box=self.myself_get_peer_apl_anchor_box(input_size,anchor_desity=long_density,aspect_ratios=self.long_horizonal_ratios,fm_sizes=fm_sizes,apl_name ='APL_05',feat_name=feat_name)
                total_apl_boxes.append(apl5_box)
        return torch.cat(total_apl_boxes,dim=0)
    def myself_get_peer_apl_anchor_box(self,input_size,anchor_desity,aspect_ratios,fm_sizes,apl_name='',feat_name=''):
        num_fms = len(self.anchor_areas)

        # fm_sizes = [(input_size/pow(2.,i+2)).ceil() for i in range(num_fms)]  # p2 -> p7 feature map sizes
        # print('------------计算标签的时候各个特征图尺寸的大小fm_sizes_lsit------',fm_sizes)
        # fm_sizes = [torch.Tensor([56., 56.]), torch.Tensor([14., 14.]), torch.Tensor([7., 7.]), torch.Tensor([4., 4.]),
        #             torch.Tensor([4., 4.]), torch.Tensor([4., 4.])]

        squre_medium_feat_map = {'feat1': 0, 'feat2': 1, 'feat3': 2, 'feat4': 3, 'feat5': 4, 'feat6': 5}
        long_feat_map={'feat2': 0, 'feat3': 1, 'feat4': 2, 'feat5': 3, 'feat6': 4}

        # print('--------------feat_name-----------',feat_name)

        which_APL=self.anchor_wh_dict[apl_name]

        # print('----------------which_apl.size()----------------:',which_APL.size())
        index=0
        if apl_name in ['APL_01','APL_02','APL_03']:
            index=squre_medium_feat_map[feat_name]
        if apl_name in ['APL_04','APL_05']: #long_anchor中没有 feat1上的
            if feat_name!='feat1':
                # print('debug(apl_name):',apl_name)
                # print('debug(feat_name):',feat_name)
                index=long_feat_map[feat_name]

        # print('-------------index--------------',index)

        wh=which_APL[index,:,:]

        # for i in range(num_fms):  # 在不同的feature上
        fm_size = fm_sizes # 每个特征图的大小（w,h）
        # print('--------------------------------------每个特征图的大小fm_size-------：',fm_size)
        grid_size = input_size / fm_size  # 原图上网格的大小
        # print('-----------------grid_size---------------', grid_size)
        fm_w, fm_h = int(fm_size[0]), int(fm_size[1])

        # ---------------------------根据传入的APL中的模块名称做垂直方向上或水平方向上的偏移----------------------#
        if apl_name in ['APL_01','APL_02','APL_04']:
            # print('---------------------------------anchor_density:',anchor_desity)
            fm_w *= int(anchor_desity)
        if apl_name in ['APL_03','APL_05']:
            # print('----------------------------------anchor-density:',anchor_desity)
            fm_h*=int(anchor_desity)

        # print('-------------------type(fm_w):', type(fm_w),'fm_w:',fm_w)
        # print('-------------------type(fm_h):', type(fm_h),'fm_h',fm_h)

        xy = meshgrid(fm_w, fm_h,anchor_desity,apl_name) + 0.5  # 得到feture上每个cell的中心坐标。




        # print('--------------meshgrid出来的xy和对应的type(xy)-------------',xy.size(),type(xy))
        xy = (xy * grid_size).view(fm_w, fm_h,1, 2).expand(fm_w, fm_h,len(aspect_ratios),
                                                                2)  # 得到原图上，每个cell中心点对应的anchor的坐标，
        # 其中同一cell所对应不同长宽比的anchor在原图坐标相同。
        # print('---得到原图上每个cell上anchor对应于原图的坐标(在特征图上做了anchor-density)---', xy.size())
        #
        # print('---------------------wh,view之前的.size()-------',wh.size())
        wh = wh.view(1, 1,len(aspect_ratios), 2).expand(fm_w, fm_h,len(aspect_ratios), 2)

        # print('---得到原图上每个cell上anchor对应的宽高（同样的在特征图上了做了anchor-density---）', wh.size())
        # print('--------------xy.size()-----------:',xy.size())
        # print('--------------wh.size()-----------:',wh.size())
        box = torch.cat([xy, wh], 2)  # [x,y,w,h]  得到原图上每个anchor的坐标和对应的宽高 size(fm_w,fm_h_len(self.aspect_ratios),4)
        # print('---得到原图上每个cell中anchor对应的坐标以及宽和高（同样在特征图上做了anchor-density----）', box.size())

        return box.view(-1, 4)# 转为 N*4






    # def _get_anchor_boxes(self, input_size):
    #     '''Compute anchor boxes for each feature map.
    #
    #     Args:
    #       input_size: (tensor) model input size of (w,h).
    #
    #     Returns:
    #       boxes: (list) anchor boxes for each feature map. Each of size [#anchors,4],
    #                     where #anchors = fmw * fmh * #anchors_per_cell
    #     '''
    #     num_fms = len(self.anchor_areas)
    #     # fm_sizes = [(input_size/pow(2.,i+2)).ceil() for i in range(num_fms)]  # p2 -> p7 feature map sizes
    #     # print('------------计算标签的时候各个特征图尺寸的大小fm_sizes_lsit------',fm_sizes)
    #     fm_sizes=[torch.Tensor([56., 56.]), torch.Tensor([14., 14.]), torch.Tensor([7., 7.]), torch.Tensor([4., 4.]), torch.Tensor([4., 4.]),torch.Tensor([4., 4.]) ]
    #     boxes = []
    #     for i in range(num_fms):#在不同的feature上
    #         fm_size = fm_sizes[i]# 每个特征图的大小（w,h）
    #         grid_size = input_size / fm_size# 原图上网格的大小
    #         print('-----------------grid_size---------------',grid_size)
    #         fm_w, fm_h = int(fm_size[0]), int(fm_size[1])
    #
    #         #---------------------------做垂直方向上的偏移----------------------#
    #         fm_w *= 2  # add vertical offset
    #
    #         #----------------------------做水平方向的偏移-----------------------#
    #         print('-------------------type(fm_w):',type(fm_w))
    #         print('-------------------type(fm_h):',type(fm_h))
    #         xy = meshgrid(fm_w,fm_h) + 0.5 #得到feture上每个cell的中心坐标。
    #         xy = (xy*grid_size).view(fm_w,fm_h,1,2).expand(fm_w,fm_h,len(self.aspect_ratios),2)#得到原图上，每个cell中心点对应的anchor的坐标，
    #         #其中同一cell所对应不同长宽比的anchor在原图坐标相同。
    #         print('---得到原图上每个cell上anchor对应于原图的坐标(在特征图上做了anchor-density)---',xy.size())
    #
    #         wh = self.anchor_wh[i].view(1,1,len(self.aspect_ratios),2).expand(fm_w,fm_h,len(self.aspect_ratios),2)
    #
    #         print('---得到原图上每个cell上anchor对应的宽高（同样的在特征图上了做了anchor-density---）',wh.size())
    #         box = torch.cat([xy,wh], 3)  # [x,y,w,h]  得到原图上每个anchor的坐标和对应的宽高 size(fm_w,fm_h_len(self.aspect_ratios),4)
    #         print('---得到原图上每个cell中anchor对应的坐标以及宽和高（同样在特征图上做了anchor-density----）',box.size())
    #         boxes.append(box.view(-1,4))# 转为 N*4
    #     return torch.cat(boxes, 0)

    def encode(self, gt_quad_boxes, labels, input_size):#input_size为(w,h)输入图片的宽和高
        # print('*************************************************进入encode---------------------')
        # print('----------------------------------------传入encode中的labels.size():',labels.size(),'labels:',labels)
        # print('----------------------------------------传入encode中的gt_quad_boxe.size():',gt_quad_boxes.size(),'gt_quad_boxes:',gt_quad_boxes)
        '''Encode target bounding boxes and class labels.

        TextBoxes++ quad_box encoder:
          tx_n = (x_n - anchor_x) / anchor_w
          ty_n = (y_n - anchor_y) / anchor_h

        Args:
          gt_quad_boxes: (tensor) bounding boxes of (xyxyxyxy), sized [#obj, 8].#每张图上的标签文本框
          labels: (tensor) object class labels, sized [#obj, ].对应的每张图上每个框的标签类别
          input_size: (int/tuple) model input size of (w,h). 图片输入的尺寸大小。
        Returns:
          loc_targets: (tensor) encoded bounding boxes, sized [#anchors,8]
          cls_targets: (tensor) encoded class labels, sized [#anchors,].
        '''
        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)

        # print('------------------传入self.myself_get_anchor_box前的input_size:',input_size)
        anchor_rect_boxes = self.myself_get_anchor_box(input_size)                 #(num_anchor, 4)
        #得到图上所有的先验anchor
        anchor_quad_boxes = change_box_order(anchor_rect_boxes, "xywh2quad")   #(num_anchor, 8)
        #将图上所有先验anchor转换格式

        # print('将原图上对应的所有框（x,y,w,h）转换为（x1,y1,x2,y2,x3,y3,x4,y4）后的tensor形状',anchor_quad_boxes.size())

        # print('encode传入的原图上文本框的坐标信息（包含8个值）的size()',gt_quad_boxes.size())

        gt_rect_boxes = change_box_order(gt_quad_boxes, "quad2xyxy")# 将标签对应的框（N*8）转化为可以计算IOU的格式
        # print('将原图对应的文本框8个坐标信息转化为（xmin,ymin,xmax,ymax）后的size():',gt_rect_boxes.size())

        time1=time.time()

        # print('-------------------------查看anchor_rect_boxes.dtpye:',anchor_rect_boxes.dtype)
        # print('-------------------------查看gt_rect_boxe.dtpye:', gt_rect_boxes.dtype)

        ious = box_iou(anchor_rect_boxes, gt_rect_boxes)# 计算原图上所有的框与原图对应的文本框标签的iou
        time2=time.time()
        # print('计算原图上所有框与原图对应文本框标签iou用时间time2-time1:',time2-time1)
        # print('所得IOU的size:',ious.size()) #N*M N表示原图上所有anchord的个数，M表示原图上拥有的文本框个数。

        #iou 形状为N*M (N表示每张图上所产生的anchor个数，M表示每张图上对应的标签框个数)
        max_ious, max_ids = ious.max(1)# max_ious表示为每个anchor 与图片文本框的所有外接矩形最大的iou值
        #max_ids 表示每个anchor所有文本框有最大iou值所对应的id
        # print('----------- max_ious-----:',max_ious)  # N(每个anchor 与所有文本外接框最大的iou值)
        # print('------------max_ious.size()-------',max_ious.size())
        # print('------------max_ids--------',max_ids)  # N(每个anchor 与文本外接框有最大iou值，所对应的id值)
        # print('------------max_ids.size()--------',max_ids.size())
        # print('------------max_ids.dtype()-------',max_ids.dtype)


        #Each anchor box matches the largest iou with the gt box


        # print('-------------------gt_quad_boxes.size():',gt_quad_boxes.size(),'max_ids.size():',max_ids.size())
        gt_quad_boxes = gt_quad_boxes[max_ids]  #(num_gt_boxes, 8) #得到每个anchor 所对应的iou最大的文本框（包含8个坐标信息）
        # print('gt_quda_boxes.size():',gt_quad_boxes.size())
        gt_rect_boxes = gt_rect_boxes[max_ids]  #(num_gt_boxes, 4)
        # print('gt_rect_boxe.size():',gt_rect_boxes.size())


        # for Rectangle boxes -> using in TextBoxes
        #gt_rect_boxes = change_box_order(gt_rect_boxes, "xyxy2xywh")
        #loc_rect_yx = (gt_rect_boxes[:, :2] - anchor_rect_boxes[:, :2]) / anchor_rect_boxes[:, 2:]
        #loc_rect_hw = torch.log(gt_rect_boxes[:, 2:] / anchor_rect_boxes[:, 2:])

        # for Quad boxes -> using in TextBoxes++
        # print('anchor_rect_boxes.size()',anchor_rect_boxes.size())
        anchor_boxes_hw = anchor_rect_boxes[:, 2:4].repeat(1, 4)
        # print('anchor_boxes_hw.size()',anchor_boxes_hw.size())

        #回归计算，每个anchor与图上标签框有最大iou框的组合（N,8）N为产生的anchor总和，即与每个anchor对应的标签框，减去anchor,再除以每个anchor的宽和高
        loc_quad_yx = (gt_quad_boxes - anchor_quad_boxes) / anchor_boxes_hw


        #loc_targets = torch.cat([loc_rect_yx, loc_rect_hw, loc_quad_yx], dim=1) # (num_anchor, 12)
        loc_targets = loc_quad_yx

        # print('lables.size():',labels.size())
        cls_targets = labels[max_ids]# 得到每个先验anchor的标签


        cls_targets[max_ious<0.5] = 0 # ignore (0.4~0.5) : -1
        # cls_targets[max_ious<0.4] = 0     # background (0.0~0.4): 0
                                         # positive (0.5~1.0) : 1
        # print('loc_targets.size():',loc_targets.size(),'cls_targets.size():',cls_targets.size())
        # print('---------------------标签制作中的cls_targets.size:',cls_targets.size())
        # print('----------------------标签制作中的cls_targets:',cls_targets)
        # print('---------------------标签制作中的cls_targets.nonzero.size():',cls_targets.nonzero().size())
        return loc_targets, cls_targets #返回每个anchor 对应于与之有最大iou值得（x1,y1,x2,y2,x3,y3,x4,y4）间的偏移和对应anchord的标签类别

    def decode(self, loc_preds, cls_preds, input_size):
        '''Decode outputs back to bouding box locations and class labels.
        Args:
          loc_preds: (tensor) predicted locations, sized [#anchors, 8].
          cls_preds: (tensor) predicted class labels, sized [#anchors, ].
          input_size: (int/tuple) model input size of (w,h).
        Returns:
          boxes: (tensor) decode box locations, sized [#obj,8].
          labels: (tensor) class labels for each box, sized [#obj,].
        '''

        input_size = torch.Tensor([input_size,input_size]) if isinstance(input_size, int) \
                     else torch.Tensor(input_size)

        print('-------------------------decode中传入self.myself_get_anchor_box中的input_size:',input_size)

        anchor_rect_boxes = self.myself_get_anchor_box(input_size).to(device)
        # print('------------------------anchro_rect_boxe.size():',anchor_rect_boxes.size())
        anchor_quad_boxes = change_box_order(anchor_rect_boxes, "xywh2quad")

        quad_boxes = anchor_quad_boxes + anchor_rect_boxes[:, 2:4].repeat(1, 4) * loc_preds  # [#anchor, 8]
        quad_boxes = torch.clamp(quad_boxes, 0, input_size[0])
        # print('----------------------------quad_boxe.size():',quad_boxes.size())
        # print('----------------------------cls_preds.size():',cls_preds.size())
        
        score, labels = cls_preds.sigmoid().max(1)       # focal loss
        #score, labels = softmax(cls_preds).max(1)          # OHEM+softmax


        # print('----------------------------score.size():',score.size())
        # print('----------------------------labels.size():',labels.size())
        # Classification score Threshold
        ids = score > self.cls_thresh # 取大于某一阈值
        ids = ids.nonzero().squeeze()   # [#obj,]
        
        score = score[ids]
        labels = labels[ids]

        # print('----------------------筛选后的score.size:',score.size)
        quad_boxes = quad_boxes[ids].view(-1, 4, 2)
        
        quad_boxes = quad_boxes.cpu().data.numpy()
        score = score.cpu().data.numpy()
        print('-----------------------筛选后的quad_boxes.size():',quad_boxes.shape)
        if len(score.shape) is 0:
            return quad_boxes, labels, score
        else:
            keep = non_max_suppression_poly(quad_boxes, score, self.nms_thresh)
            return quad_boxes[keep], labels[keep], score[keep]

def debug(): 
    encoder = DataEncoder()

    anchor_wh = encoder._get_anchor_wh()

    input_size = 32
    input_size =  torch.Tensor([input_size,input_size])
    anchor = encoder._get_anchor_boxes(input_size)

    print("anchor.size() : ", anchor.size())
    for i in anchor:
        print(i)
    exit()
    test = torch.randn((3, 8))
    #test2 = torch.reshape(test, (-1, 4, 2))
    test2 = test.view((-1, 4, 2))
    print("test : ", test, test.size())
    print("test2 : ", test2, test2.size())

    gt_quad_boxes = torch.randn((41109, 8))
    labels = torch.randn((41109, 1))
    result_encode = encoder.encode(gt_quad_boxes, labels, input_size)
    print(result_encode[0].size())
    print(result_encode[1].size())


def mydebug():
    anchor_wh={}
    encoder=DataEncoder()
    input_size = 224
    input_size = torch.Tensor([input_size, input_size])

    print('-----input_size-----:',input_size)

    anchor = encoder.myself_get_anchor_box(input_size)


    print('anchor.size():',anchor.size())

    # print(anchor_wh.keys())
    # print('-----------------anchor_wh------------------',len(anchor_wh))
    # print(anchor_wh.values())

    gt_quad_boxes = torch.randn((11, 8))
    labels = torch.randn((11, 1))
    result_encode = encoder.encode(gt_quad_boxes, labels, input_size)
    print(result_encode[0].size())
    print(result_encode[1].size())




