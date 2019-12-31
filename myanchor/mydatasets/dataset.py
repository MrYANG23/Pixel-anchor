import torch
import math
import os
import PIL
import numpy as np
import cv2
import torchvision.transforms as transforms
from shapely.geometry import Polygon
from torch.utils import data
from PIL import Image
from utils.anchorutils import meshgrid, box_iou, change_box_order, softmax
from utils.anchorencoder import DataEncoder


#-----------pixel-dataset-function---------
from utils.pixelutils import extract_vertices,adjust_height,rotate_img,crop_img,get_score_geo,new_crop_img


#-----------anchor-dataset-function--------
from utils.anchor_augmentations import Augmentation_traininig,ToCV2Image
#-----------------mydataset------------------


class custom_dataset(data.Dataset):
    def __init__(self,img_path,gt_path,pixel_scale=0.25,lenght=640):
        super(custom_dataset,self).__init__()
        self.img_list=[os.path.join(img_path, img_file) for img_file in sorted(os.listdir(img_path))]
        self.gt_file=[os.path.join(gt_path, gt_file) for gt_file in sorted(os.listdir(gt_path))]
        # for i in self.img_list:
        #     print('-----i:',i)
        # for j in self.gt_file:
        #     print('-----j:',j)
        # # print('--------------',self.img_list)
        # # print('--------------',self.gt_file)
        # exit()
        self.pixel_scale=pixel_scale
        self.lenght=lenght
        self.number=len(self.img_list)
        self.encoder=DataEncoder()
        self.transform=Augmentation_traininig
    def __getitem__(self,index ):
        with open(self.gt_file[index], 'r') as f:
            lines = f.readlines()
        vertices_0, labels_0 = extract_vertices(lines)
        # print('----------------------从图片行读取每张图上所有框，未做处理前的坐标vertices_0:',vertices_0)
        # print('----------------------从图片行读取每张图上所有框，未做处理前的标签labels_0:', labels_0)
        # print('vertices_0.shape:', vertices_0.shape)
        # print('labels_0.shape:', labels_0.shape)
        img_0 = Image.open(self.img_list[index])
        # img2 = np.array(img_0)
        #-------------------------------只做图片的resize---------------------
        anchor_img, boxes, labels = self.transform(size=self.lenght)(img_0, vertices_0,labels_0)#经过pytorch中的transforms后图片通道数会变为c.h,w
        # print('--------------------------anchor_img.size():',anchor_img.size())
        # print('----------------------type(anchor_img.numpy()):',type(anchor_img.numpy()))

        # img = Image.fromarray(anchor_img.permute(1,2,0).numpy(),mode="RGB")#从tensor转为numpy再转为PIL


        #-------------------------------pixel部分图片与标签预处理------------------------------
        # img1, vertices1 = adjust_height(img, vertices)
        # img1, vertices1 = rotate_img(img1, vertices1)
        # img, vertices = new_crop_img(img_0, vertices_0, labels_0, self.lenght)
        # print('---------------经过pixel部分图片预处理后vertice.shape---------:', vertices.shape)
        # print('经过pixel部分，图片预处理后的图片大小img.shape:',np.array(img).shape)
        # print('传入get_score_geo前的vertices.shape:', vertices.shape)



        #--------------------------------------------pixel部分制作标签---------------------
        # score_map, geo_map, ignored_map = get_score_geo(img, boxes.numpy(), labels.numpy(), self.pixel_scale, self.lenght)
        #------------------anchor部分标签---------------
        size = self.lenght

        # inputs = torch.zeros(1, 3, size, size)
        # img2=np.array(img_0)
        #------------------------修改anchor部分预处理---------
        # boxes=vertices
        # boxes=torch.Tensor(vertices)
        # print('--------------------------修改anchor部分预处理后boxes.size():',boxes.size())
        # labels=torch.Tensor(labels_0).type(torch.long)
        # print('--------------------------修改anchor部分预处理后labels.size():',labels.size())


        #---------------------------------anchor部分预处理-----------------------------
        # anchor_img, boxes, labels = self.transform(size=size)(img2, vertices_0,labels_0)  # 对传入的图片，每张图上对应的文本框，以及文本框对应的类别做预处理。
        # print('经过anchor部分预处理后anchor_img.size():',anchor_img.size())
        # print('经过anchor部分预处理后的boxes.size():', boxes.size(),'type(boxes)',type(boxes))
        # print('经过anchor部分预处理后的后的labels.size():', labels.size(),'type(labels)',type(labels))

        # inputs = img2
        # img_dir = self.img_list[index]
        # print('将对应的标签文本框，和对应类别标签传入encode模块，：')

        #----------------------------------anchor部分制作标签------------------------------
        loc_target, cls_target = self.encoder.encode(boxes.type(torch.float32), labels,
                                                     input_size=(size, size))
       #返回每张图上产生的先验anchor与之有最大iou标签框间相对于每个anchor的（w,h）间的偏移和每个anchor对应的标签。
        # print('--------------------------------------------dataset中传出的torch.sum(cls_target):',torch.sum(cls_target))
        return anchor_img, loc_target,cls_target
    def __len__(self):
        return self.number
#------------------------------测试mydataset--------------------------
def test():
	print('--------------jinrutest()----------')
	img_path='/data/yanghan/ICDAR2015/ch4_training_images'
	img_lablepath='/data/yanghan/ICDAR2015/ch4_training_localization_transcription_gt'
	mydataset = custom_dataset(img_path, img_lablepath)
	dataloader = torch.utils.data.DataLoader(mydataset, batch_size=2, shuffle=False, num_workers=0)
	print('len(dataloader):', len(dataloader))
	# for n,(pixel_list,anchor_list) in enumerate(dataloader):
    #     print('n:',n)
    #     print('len(pixel_list):',len(pixel_list))
    #     exit()
	for n,(pixel_list,anchor_list) in enumerate(dataloader):
		print('len(pixel_list):',len(pixel_list))
		print('len(anchor_list):',len(anchor_list))
		break
# if __name__ == '__main__':
#     test()


