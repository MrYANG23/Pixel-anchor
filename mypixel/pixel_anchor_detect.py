import lanms
import numpy as np
import torch
from torchvision import transforms
from PIL import Image,ImageDraw
from utils.pixelutils import get_rotate_mat
import os
from utils.detect_utils import plot_boxes,load_pil,get_boxes,resize_img,detect,adjust_ratio
from utils.anchorencoder import DataEncoder


def pixelandanchor_detect(img,model,device,img_save_path,img_size=640):
    ' img PIL读取出来的图片'
    'model 加载权重后网络'
    'device 指定用几号gpu，还是cpu'
    width,height=img.size[0],img.size[1]
    orignal_img=img# 原图大小
    img=img.resize((img_size,img_size))# 改变推理过程中输入图片的大小
    img,ratio_h,ratio_w=resize_img(img)#   pixel中将图片转为可以整除32的长和边
    img_input=load_pil(img).to(device)
    #-------------------------pixel_detect部分----------------
    with torch.no_grad():
        score, geo,attention_map= model(img_input)
        print('-----------------------从网络输出的score.shape:',score.shape)
    pixel_boxes=get_boxes(score.squeeze(0).cpu().numpy(),geo.squeeze(0).cpu().numpy())
    print('----------------------pixel_boxes.shape:',pixel_boxes.shape)

    #------------------------------新加功能-------------------
    pixel_boxes[:, [0,1,2,3,4,5,6,7]]/=img_size
    pixel_boxes[:, [0, 2, 4, 6]] *= width  # 将改变size后所得到的推理框还原到原图的尺寸大小
    pixel_boxes[:, [1, 3, 5, 7]] *= height  # 同上
    #------------------------------新加功能-------------------


    pixel_boxes_finals=adjust_ratio(pixel_boxes,ratio_w,ratio_h)
    print('----------------------pixel_boxes_finals.shape:',pixel_boxes_finals.shape)
    plot_img = plot_boxes(orignal_img, pixel_boxes_finals)

    plot_img.save(img_save_path)


from model.backbone import PixelAnchornet

if __name__ == '__main__':
    img_path='/data/yanghan/ICDAR2015/ch4_test_images/img_242.jpg'
    pth_path='./pths/model_epoch_400.pth'
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
    # img = img.resize((640, 640))
    print('-----------------传入图片的img.size:',img.size)# 1280,720
    pixelandanchor_detect(img,net,device,image_save_path+'{}'.format('/12_26_epoch400_test_img242_size640_return.jpg'))













