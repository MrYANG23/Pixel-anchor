import lanms
import numpy as np
import torch
from torchvision import transforms
from PIL import Image,ImageDraw
from utils.pixelutils import get_rotate_mat
import os
from utils.detect_utils import plot_boxes,load_pil,get_boxes,resize_img,detect,adjust_ratio
from utils.anchorencoder import DataEncoder


def pixelandanchor_detect(img,model,device,img_save_path,size_scale=640):
    ' img PIL读取出来的图片'
    'model 加载权重后网络'
    'device 指定用几号gpu，还是cpu'
    orignal_img=img
    width, height = img.size[0],img.size[1]  # 取得原图的宽和高
    img = img.resize((size_scale, size_scale))# 变为与训练的尺寸一致，能得到更好的效果
    img,ratio_h,ratio_w=resize_img(img)#   pixel中将图片转为可以整除32的长和边
    #——--------------------------------------修改12.24---------------

    #----------------------------------------修改12.24---------------

    img_input=load_pil(img).to(device) #   load_pil 加载pil读取的图片，并做初步的预处理
    #-------------------------pixel_detect部分----------------
    with torch.no_grad():
        pre_location, pre_class = model(img_input)
        print('-------------------pre_class-------------:',pre_class)

        # print('-----------------------从网络输出的score.shape:',score.shape)
    # pixel_boxes=get_boxes(score.squeeze(0).cpu().numpy(),geo.squeeze(0).cpu().numpy())
    # print('----------------------pixel_boxes.shape:',pixel_boxes)
    # pixel_boxes_finals=adjust_ratio(pixel_boxes,ratio_w,ratio_h)
    # print('----------------------pixel_boxes_finals:',pixel_boxes_finals)
    # plot_img = plot_boxes(img, pixel_boxes_finals)
    #
    # plot_img.save(img_save_path)

    #--------------------------anchor_detect部分---------------
    decoder=DataEncoder()

    print('---------------------pre_location.size():',pre_location.size())
    print('---------------------pre_class.size():',pre_class.size())
    print('------------------传入decode的img.size():',img.size)
    boxes, labels, scores =decoder.decode(pre_location.data.squeeze(0),pre_class.data.squeeze(0),img.size)

    #---------------------------修改12.24----------------------
    boxes /= size_scale  # 除以所设置的图片size的的大小
    boxes *= ([[width, height]] * 4)  # 还原为原图上的框



    draw = ImageDraw.Draw(orignal_img)
    print('------------------------adjust_ratio之前的boxes.shape:',boxes.shape)


    boxes = adjust_ratio(boxes.reshape(-1,8),ratio_w,ratio_h)
    print('------------------------adjust_ratio之后的boxes.shape:',boxes.shape)

    boxes = boxes.reshape(-1, 4, 2)

    for box in boxes:
        draw.polygon(np.expand_dims(box, 0), outline=(0, 255, 0))

    orignal_img.save(img_save_path)

from model.backbone import PixelAnchornet

if __name__ == '__main__':
    img_path='/data/yanghan/ICDAR2015/ch4_test_images/img_242.jpg'
    pth_path='/tmp/myanchor/new_pths/model_epoch_25.pth'
    image_save_path='./out_test_images'
    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)
    device=torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print('----------------------------------')
    net = PixelAnchornet(pretrained=False).to(device)
    model_checkpoint=torch.load(pth_path)
    net.load_state_dict(model_checkpoint)
    net.eval()
    img=Image.open(img_path)
    # img=img.resize((640,640))
    print('-----------------传入图片的img.size:',img.size)# 1280,720
    pixelandanchor_detect(img,net,device,image_save_path+'{}'.format('/12_27_epoch_newpath25_test_imag242__size640_return_orignal.jpg'))














