import torch
import torch.nn as nn
from model.resnet50 import Resnet
class APL(nn.Module):
    def __init__(self,features_name,inplanes,anchor_density_medium_list=[1,2,3,4,3,2],anchor_density_long_list=[4,4,6,4,3]):
        super(APL,self).__init__()
        self.inplanes=inplanes
        self.anchor_density_medium_list=anchor_density_medium_list
        self.anchor_density_long_list=anchor_density_long_list

        self.features_name = features_name

        self.squre_density_num=1

        #--------------------确定APL中不同卷积核对应的不同框的个数------------------
        if  self.features_name == 'feat1':# feat1中没有long_anchor
            self.medium_density_num=self.anchor_density_medium_list[0]

        elif self.features_name =='feat2':
            self.medium_density_num = self.anchor_density_medium_list[1]
            self.long_density_num = self.anchor_density_long_list[0]

        elif self.features_name =='feat3':
            self.medium_density_num = self.anchor_density_medium_list[2]
            self.long_density_num = self.anchor_density_long_list[1]

        elif self.features_name=='feat4':
            self.medium_density_num = self.anchor_density_medium_list[3]
            self.long_density_num = self.anchor_density_long_list[2]
        elif self.features_name=='feat5':
            self.medium_density_num = self.anchor_density_medium_list[4]
            self.long_density_num = self.anchor_density_long_list[3]
        elif self.features_name=='feat6':
            self.medium_density_num = self.anchor_density_medium_list[5]
            self.long_density_num = self.anchor_density_long_list[4]





        self.square_conv_loc=nn.Conv2d(in_channels=self.inplanes,out_channels=8*self.squre_density_num,kernel_size=(3,5),padding=(1,2),stride=1)
        self.square_conv_cla=nn.Conv2d(in_channels=self.inplanes,out_channels=1*self.squre_density_num,kernel_size=(3,5),padding=(1,2),stride=1)


        self.medium_vertical_loc=nn.Conv2d(in_channels=self.inplanes,out_channels=32*self.medium_density_num,kernel_size=(5,3),padding=(2,1),stride=1)
        self.medium_vertical_cla=nn.Conv2d(in_channels=self.inplanes,out_channels=4*self.medium_density_num,kernel_size=(5,3),padding=(2,1),stride=1)

        self.medium_horizonal_loc=nn.Conv2d(in_channels=self.inplanes,out_channels=32*self.medium_density_num,kernel_size=(3,5),padding=(1,2),stride=1)
        self.medium_horizonal_cla=nn.Conv2d(in_channels=self.inplanes,out_channels=4*self.medium_density_num,kernel_size=(3,5),padding=(1,2),stride=1)

        if features_name=='feat2':
            self.long_vertical_loc=nn.Conv2d(in_channels=self.inplanes,out_channels=24*self.long_density_num,kernel_size=(33,1),padding=(16,0),stride=1)
            self.long_vertical_cla=nn.Conv2d(in_channels=self.inplanes,out_channels=3*self.long_density_num,kernel_size=(33,1),padding=(16,0),stride=1)

            self.long_horizonal_loc=nn.Conv2d(in_channels=self.inplanes,out_channels=24*self.long_density_num,kernel_size=(1,33),padding=(0,16),stride=1)
            self.long_horizonal_cla=nn.Conv2d(in_channels=self.inplanes,out_channels=3*self.long_density_num,kernel_size=(1,33),padding=(0,16),stride=1)


        if features_name=='feat3':
            self.long_vertical_loc=nn.Conv2d(in_channels=self.inplanes,out_channels=24*self.long_density_num,kernel_size=(29,1),padding=(14,0),stride=1)
            self.long_vertical_cla=nn.Conv2d(in_channels=self.inplanes,out_channels=3*self.long_density_num,kernel_size=(29,1),padding=(14,0),stride=1)

            self.long_horizonal_loc=nn.Conv2d(in_channels=self.inplanes,out_channels=24*self.long_density_num,kernel_size=(1,29),padding=(0,14),stride=1)
            self.long_horizonal_cla=nn.Conv2d(in_channels=self.inplanes,out_channels=3*self.long_density_num,kernel_size=(1,29),padding=(0,14),stride=1)
        if features_name=='feat4':
            self.long_vertical_loc=nn.Conv2d(in_channels=self.inplanes,out_channels=24*self.long_density_num,kernel_size=(15,1),padding=(7,0),stride=1)
            self.long_vertical_cla=nn.Conv2d(in_channels=self.inplanes,out_channels=3*self.long_density_num,kernel_size=(15,1),padding=(7,0),stride=1)

            self.long_horizonal_loc=nn.Conv2d(in_channels=self.inplanes,out_channels=24*self.long_density_num,kernel_size=(1,15),padding=(0,7),stride=1)
            self.long_horizonal_cla=nn.Conv2d(in_channels=self.inplanes,out_channels=3*self.long_density_num,kernel_size=(1,15),padding=(0,7),stride=1)


        if features_name=='feat5':
            self.long_vertical_loc = nn.Conv2d(in_channels=self.inplanes, out_channels=24*self.long_density_num, kernel_size=(15, 1), padding=(7,0),stride=1)
            self.long_vertical_cla=nn.Conv2d(in_channels=self.inplanes, out_channels=3*self.long_density_num, kernel_size=(15, 1), padding=(7,0),stride=1)


            self.long_horizonal_loc = nn.Conv2d(in_channels=self.inplanes, out_channels=24*self.long_density_num, kernel_size=(1, 15), padding=(0,7),stride=1)
            self.long_horizonal_cla=nn.Conv2d(in_channels=self.inplanes, out_channels=3*self.long_density_num, kernel_size=(1, 15), padding=(0,7),stride=1)

        if features_name=='feat6':
            self.long_vertical_loc = nn.Conv2d(in_channels=self.inplanes, out_channels=24*self.long_density_num, kernel_size=(15, 1), padding=(7,0),stride=1)
            self.long_vertical_cla=nn.Conv2d(in_channels=self.inplanes, out_channels=3*self.long_density_num, kernel_size=(15, 1), padding=(7,0),stride=1)


            self.long_horizonal_loc = nn.Conv2d(in_channels=self.inplanes, out_channels=24*self.long_density_num, kernel_size=(1, 15),padding=(0,7), stride=1)
            self.long_horizonal_cla=nn.Conv2d(in_channels=self.inplanes, out_channels=3*self.long_density_num, kernel_size=(1, 15),padding=(0,7), stride=1)




    def forward(self,x):
        location=[]
        classify=[]

        # print('--------传入APL模块的特征图size（）-----------',x.size())
        square_conv_loc=self.square_conv_loc(x)
        # print('--------经过APL模块中square_conv_loc后的size（）----------',square_conv_loc.size())
        square_conv_loc=square_conv_loc.permute(0,2,3,1).contiguous().view(x.size(0),-1,8)
        location.append(square_conv_loc)

        square_conv_cla=self.square_conv_cla(x)
        square_conv_cla=square_conv_cla.permute(0,2,3,1).contiguous().view(x.size(0),-1,1)
        classify.append(square_conv_cla)


        medium_vertical_loc=self.medium_vertical_loc(x)
        # print('--------经过APL模块中medium_vertical_loc后的size（）----------', medium_vertical_loc.size())
        medium_vertical_loc=medium_vertical_loc.permute(0,2,3,1).contiguous().view(x.size(0),-1,8)

        location.append(medium_vertical_loc)

        medium_vertical_cla=self.medium_vertical_cla(x)
        medium_vertical_cla=medium_vertical_cla.permute(0,2,3,1).contiguous().view(x.size(0),-1,1)

        classify.append(medium_vertical_cla)


        medium_horizonal_loc=self.medium_horizonal_loc(x)
        # print('--------经过APL模块中medium_horizonal_loc后的size（）----------', medium_horizonal_loc.size())
        medium_horizonal_loc=medium_horizonal_loc.permute(0,2,3,1).contiguous().view(x.size(0),-1,8)
        location.append(medium_horizonal_loc)

        medium_horizonal_cla=self.medium_horizonal_cla(x)
        medium_horizonal_cla=medium_horizonal_cla.permute(0,2,3,1).contiguous().view(x.size(0),-1,1)
        classify.append(medium_horizonal_cla)

        if self.features_name !='feat1':

            long_vertical_loc=self.long_vertical_loc(x)
            # print('--------经过APL模块中long_vertical_loc后的size（）----------', long_vertical_loc.size())
            long_vertical_loc=long_vertical_loc.permute(0,2,3,1).contiguous().view(x.size(0),-1,8)

            location.append(long_vertical_loc)

            long_vertical_cla=self.long_vertical_cla(x)
            long_vertical_cla=long_vertical_cla.permute(0,2,3,1).contiguous().view(x.size(0),-1,1)
            classify.append(long_vertical_cla)

            long_horizonnal_loc=self.long_horizonal_loc(x)
            # print('--------经过APL模块中long_horizonal_loc后的size（）----------', long_horizonnal_loc.size())
            long_horizonnal_loc=long_horizonnal_loc.permute(0,2,3,1).contiguous().view(x.size(0),-1,8)
            location.append(long_horizonnal_loc)

            long_horizonnal_cla=self.long_horizonal_cla(x)
            long_horizonnal_cla=long_horizonnal_cla.permute(0,2,3,1).contiguous().view(x.size(0),-1,1)
            classify.append(long_vertical_cla)

        fms_loc=torch.cat(location,dim=1)
        fms_cla=torch.cat(classify,dim=1)

        return fms_loc,fms_cla



class Anchornet(nn.Module):
    def __init__(self,inplanes):#inplanes 1/4features上的通道数
        print('----------------进入Anchornet-------------')
        super(Anchornet,self).__init__()
        self.inplanes=inplanes
        self.apl1=APL('feat1',self.inplanes)# 从特征图1中的到的回归信息和分类信息

        self.apl2=APL('feat2',self.inplanes*4)#从特征图2中的到的回归信息和分类信息
        self.apl3=APL('feat3',self.inplanes)#从特征图3中的到的回归信息和分类信息
        self.apl4=APL('feat4',self.inplanes)#从特征图4中的到的回归信息和分类信息
        self.apl5=APL('feat5',self.inplanes)#从特征图5中的到的回归信息和分类信息
        self.apl6=APL('feat6',self.inplanes)#从特征图6中的到的回归信息和分类信息

        self.conv_1=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1,stride=1)
        self.conv_2=nn.Conv2d(in_channels=1024,out_channels=1024,kernel_size=3,padding=1,stride=1)

        self.conv1=nn.Conv2d(in_channels=1024,out_channels=256,kernel_size=3,padding=1,stride=2)
        self.relu1=nn.ReLU()

        self.conv2=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=1,stride=2)
        self.relu2=nn.ReLU()

        self.atorus1=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=2,dilation=2)
        self.atorus_relu1=nn.ReLU()
        self.atorus2=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=2,dilation=2)
        self.atorus_relu2=nn.ReLU()

    def forward(self,one_div4,one_div16,attentionmap):

        fx_location=[]
        fx_classify=[]
        attention_exp=torch.exp(attentionmap)
        ap1_input=torch.mul(one_div4,attention_exp)
        print('----------------------------one_div4.size():',one_div4.size())
        apl1_loc,apl1_cla=self.apl1(self.relu1(self.conv_1(self.relu1(self.conv_1(ap1_input)))))#加了2层卷积核激活函数
        # print('----------apl1_loc.sum()-------:',apl1_loc.sum())
        # print('----------apl1_cla.sum()-------:',apl1_cla.sum())
        fx_location.append(apl1_loc)
        fx_classify.append(apl1_cla)
        print('-----------------------------one_div16.size():',one_div16.size())
        apl2_loc,apl2_cla=self.apl2(self.relu2(self.conv_2(one_div16)))
        # print('-----------apl2_loc.sum()-------:',apl2_loc.sum())
        # print('-----------apl2_cla.sum()-------:',apl2_cla.sum())
        fx_location.append(apl2_loc)
        fx_classify.append(apl2_cla)
        conv1=self.conv1(one_div16)
        print('----------------conv1.size():',conv1.size())
        apl3_loc,apl3_cla=self.apl3(self.relu1(conv1))
        # print('------------apl3_loc.sum()------:',apl3_loc.sum())
        # print('------------apl3_cla.sum()------:',apl3_cla.sum())
        fx_location.append(apl3_loc)
        fx_classify.append(apl3_cla)


        conv2=self.conv2(self.relu1(conv1))
        print('---------------------conv2.size()----------------:',conv2.size())
        apl4_loc,apl4_cla=self.apl4(self.relu2(conv2))
        # print('-------------apl4_loc.sum()-----:',apl4_loc.sum())
        # print('-------------apl4_cla.sum()-----:',apl4_cla.sum())

        fx_location.append(apl4_loc)
        fx_classify.append(apl4_cla)


        atorus1=self.atorus1(self.relu2(conv2))
        print('----------------------atorus1.size()--------------:',atorus1.size())
        apl5_loc,apl5_cla=self.apl5(self.atorus_relu1(atorus1))
        # print('-------------apl5_loc.sum()-----:', apl5_loc.sum())
        # print('-------------apl5_cla.sum()-----:', apl5_cla.sum())
        fx_location.append(apl5_loc)
        fx_classify.append(apl5_cla)

        atorus2=self.atorus2(self.atorus_relu1(atorus1))
        print('-----------------------atorus2.size()--------------:',atorus2.size())
        apl6_loc,apl6_cla=self.apl6(self.atorus_relu2(atorus2))
        # print('-------------apl6_loc.sum()-----:', apl6_loc.sum())
        # print('-------------apl6_cla.sum()-----:', apl6_cla.sum())
        fx_location.append(apl6_loc)
        fx_classify.append(apl6_cla)

        return torch.cat(fx_location,dim=1) ,torch.cat(fx_classify,dim=1)

if __name__ == '__main__':
    import torch
    #--------------------------------测试APL中的卷积核细节---------------------
    # a=torch.randn(1,256,224,224)
    # conv1=nn.Conv2d(in_channels=3,out_channels=8,kernel_size=(3,5),padding=(1,2),stride=1)
    # conv2=nn.Conv2d(in_channels=3,out_channels=32,kernel_size=(5,3),padding=(2,1),stride=1)
    # conv3=nn.Conv2d(in_channels=3,out_channels=24,kernel_size=(33,1),padding=(16,0),stride=1)
    # conv4=nn.Conv2d(in_channels=3,out_channels=24,kernel_size=(1,33),padding=(0,16),stride=1)
    # conv5=nn.Conv2d(in_channels=3,out_channels=24,kernel_size=(29,1),padding=(14,0),stride=1)
    # conv6=nn.Conv2d(in_channels=3,out_channels=24,kernel_size=(1,29),padding=(0,14),stride=1)
    # conv7=nn.Conv2d(in_channels=3,out_channels=24,kernel_size=(15,1),padding=(7,0),stride=1)
    # conv8= nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(1, 15), padding=(0,7),stride=1)
    #
    # conv9=nn.Conv2d(in_channels=256,out_channels=3,kernel_size=3,padding=1,stride=2)
    # conv10=nn.Conv2d(in_channels=256,out_channels=256,kernel_size=3,padding=2,dilation=2)

    # b=conv10(a)
    # print(b.size())
    #--------------------测试APL--------------
    # apl1=APL('feat2',256)
    # fms_loc,fm_cla=apl1(a)
    # print('type(fms_loc):',type(fms_loc))
    # print('fms_loc.size():',fms_loc.size())
    #
    # print('type(fm_cla):',type(fm_cla))
    # print('fm_cla.size():',fm_cla.size())

    #---------------------测试anchornet------------
    anchornet=Anchornet(1024)

    one_div4=torch.randn(1,256,56,56)
    attention_map=one_div4
    one_div16=torch.randn(1,256,14,14)
    loc,cla=anchornet(one_div4,one_div16,attention_map)
    print('loc.size:',loc.size())
    print('cla.size:',cla.size())























