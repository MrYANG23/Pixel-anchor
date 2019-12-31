import torch
import torch.nn as nn
import torch.functional as F


class ASPP(nn.Module):
    def __init__(self, in_channel=1024, depth=256):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        # self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block3=nn.Conv2d(in_channel,depth,3,1,padding=3,dilation=3)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block9=nn.Conv2d(in_channel,depth,3,1,padding=9,dilation=9)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block15=nn.Conv2d(in_channel,depth,3,1,padding=15,dilation=15)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        # self.conv_1x1_output = nn.Conv2d(depth * 8, depth, 1, 1)


    def forward(self, x):
        print('------------------------pixelnet中asspnet模块输入的x.sum():',x.sum())
        size = x.shape[2:]
        # image_features = self.mean(x)
        # image_features = self.conv(image_features)
        # print('first_image_features.shape:',image_features.shape)
        # image_features =nn.functional.interpolate(image_features, size=size, mode='bilinear',align_corners=True)
        # print('image_Features.shape:',image_features.shape)

        # atrous_block1 = self.atrous_block1(x)
        # print('atrous_block1.size()',atrous_block1.size())

        atrous_block3=self.atrous_block3(x)
        # print('atrous_block3.size():',atrous_block3.size())

        atrous_block6 = self.atrous_block6(x)
        # print('atrous_block6.size()',atrous_block6.shape)

        atrous_block9=self.atrous_block9(x)
        # print('atrous_block9.size()', atrous_block9.shape)

        atrous_block12 = self.atrous_block12(x)
        # print('atrous_block12.size():',atrous_block12.size())

        atrous_block15=self.atrous_block15(x)
        # print('atrous_block15.size()', atrous_block15.shape)

        atrous_block18 = self.atrous_block18(x)
        # print('atrous_block18.size()', atrous_block18.shape)

        net =torch.cat([atrous_block3, atrous_block6,atrous_block9,atrous_block12, atrous_block15,atrous_block18],dim=1)
        return net
if __name__ == '__main__':

    # a=torch.randn(1,3,224,224)
    # print(a.shape)
    # print(a.size())
    # b=a.shape[2:]
    # print(b)

    #--------------------测试ASSP模块-----------------
    b=torch.randn((1,512,224,224))
    net=ASPP()
    outtensor=net(b)
    print('----------outtensor.shape---------',outtensor.shape)










