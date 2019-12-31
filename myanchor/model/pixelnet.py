import torch
import torch.nn as nn
from model.ASSPnet import ASPP
from model.resnet50 import Resnet
import model.resnet50
import math


class Rboexs_predictor(nn.Module):
    def __init__(self):
        super(Rboexs_predictor,self).__init__()
        self.conv1 = nn.Conv2d(256, 1, 1)
        self.sigmoid1 = nn.Sigmoid()
        self.conv2 = nn.Conv2d(256, 4, 1)
        self.sigmoid2 = nn.Sigmoid()
        self.conv3 = nn.Conv2d(256, 1, 1)
        self.sigmoid3 = nn.Sigmoid()
        self.scope = 640
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self,x):
        score = self.sigmoid1(self.conv1(x))
        loc = self.sigmoid2(self.conv2(x)) * self.scope
        # loc = self.sigmoid2(self.conv2(x))
        angle = (self.sigmoid3(self.conv3(x)) - 0.5) * math.pi
        geo = torch.cat((loc, angle), 1)
        # print('---------------------------------从pixel_net网络中出来的socre.size():',score.size())
        # print('---------------------------------从pixel_net网络中出来的socre:',score)
        return score, geo


class Attention_map(nn.Module):
    def __init__(self):
        super(Attention_map,self).__init__()
        self.conv=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1)

    def forward(self,x):
        attention_map=self.conv(x)

        return attention_map


class Pixelnet(nn.Module):
    def __init__(self):
        print('-------------------------------------进入Pixelnet-----------------------------')
        super(Pixelnet,self).__init__()
        self.asspnet=ASPP()

        self.conv1=nn.Conv2d(in_channels=1536,out_channels=256,kernel_size=3,padding=1,stride=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu1 = nn.Sigmoid()
        self.conv2=nn.Conv2d(in_channels=768,out_channels=256,kernel_size=3,padding=1,stride=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.relu2 = nn.Sigmoid()

        self.conv3=nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,padding=1,stride=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.Sigmoid()
        self.robxe=Rboexs_predictor()
        self.attention_map=Attention_map()


    def forward(self,features_list):
        size1=features_list[0].shape[2:]
        size2=features_list[1].shape[2:]

        # print('----------------------传入asspnet模块中的features_list[0]')
        # print('-------------features_lsit[2].size()------：',features_list[2].size())
        asspnet=self.asspnet(features_list[2])
        # print('----------------------pixlenet中pixel部分中asspnet.sum():',asspnet.sum())

        conv1=self.relu1(self.bn1(self.conv1(asspnet)))


        # print('------------------------pixlenet中pixel部分中conv1.sum():',conv1.sum())
        upsample1=nn.functional.interpolate(conv1,size=size2,mode='bilinear',align_corners=True)
        # print('upsample1.size():',upsample1.size())
        # print('faatures_list[1].size():',features_list[1].size())


        conv2=self.relu2(self.bn2(self.conv2(torch.cat([features_list[1],upsample1],dim=1))))
        # print('------------------------pixelnet中pixel部分中conv2.sum():',conv2.sum())
        upsample2=nn.functional.interpolate(conv2,size=size1,mode='bilinear',align_corners=True)
        # print('upsampel2.size():',upsample2.size())


        conv3=self.bn3(self.conv3(torch.cat([upsample2,features_list[0]],dim=1)))
        # print('--------------------------pixelnet中pixel部分中conv3.sum():',conv3.sum())
        # print('分别输给Rbox和attention_map的特征图尺寸大小，conv3.size()',conv3.size())

        score,geo=self.robxe(conv3)
        # print('---------------------------pixelnet中pixel部分最终输出score的socre.size():',score.size())
        # print('---------------------------pixelnet中pixel部分最终输出score的torch.sum(score):',torch.sum(score))
        # print('---------------------------pixelnet中pixel部分最终输出的score：',score)

        attention_map=self.attention_map(conv3)
        return score,geo,attention_map
if __name__ == '__main__':
    mylist=[]

    #-------------------------pixelnet网络测试---------#
    #------网络的输出尺寸大小为原图的1/4，输入的尺寸分别为原图的1/4,1/8,1/16-----#

    a=torch.randn(1,256,224,224)
    mylist.append(a)
    b=torch.randn(1,256,56,56)
    mylist.append(b)
    c=torch.randn(1,512,28,28)
    mylist.append(c)
    # print(mylist)

    net=Pixelnet()
    out,attention_map=net(mylist)
    score=out[0]
    geo=out[1]
    print('Pixelnet输出的结果为：',type(out))
    print('输出的score的size()为：',score.size())
    print('输出的geo的size()为：',geo.size())
    print('attention_map.size:',attention_map.size())


























