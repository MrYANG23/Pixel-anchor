import lanms
import numpy as np
import torch
from torchvision import transforms
from PIL import Image,ImageDraw
from utils.pixelutils import get_rotate_mat
import os
from utils.anchor_nms_poly import non_max_suppression_poly
from utils.detect_utils import plot_boxes,load_pil,get_boxes,resize_img,detect,adjust_ratio
from utils.anchorencoder import DataEncoder


def pixle_anchor_detect(img,model,device,NMS_choice='',NMS_thresh=0.1,img_save_pths='',img_size=640): #改变推理图输入大小，得到anchor部分和pixel部分的框后一起做NMS
    orignal_img=img
    width,heigth=img.size[0],img.size[1]
    img=img.resize((img_size,img_size))# 改为与训练图片尺寸一样大小

    img,ratio_h,ratio_w=resize_img(img)#   pixel中将图片转为可以整除32的长和边
    img_input=load_pil(img).to(device)
    #-------------------------pixel_detect部分----------------
    with torch.no_grad():
        score, geo, attention_map,pre_location, pre_class = model(img_input)

    pixel_boxes=get_boxes(score.squeeze(0).cpu().numpy(),geo.squeeze(0).cpu().numpy(),if_eval=True)#
    #设置为eval为true的时候，都不经过各自的nms

    print('----------------------pixel_boxes.shape:',pixel_boxes.shape)# (n, 9)

    #--------------------------anchor_detect部分---------------
    decoder=DataEncoder()
    anchor_boxes, labels, scores =decoder.decode(pre_location.data.squeeze(0),pre_class.data.squeeze(0),img.size,if_eval=True)
    #得到的为经过阈值筛选后的，并没有经过NMS

    if pixel_boxes is None and anchor_boxes is None:
        return pixel_boxes
    if pixel_boxes is None and anchor_boxes is not None:
        score = np.expand_dims(scores, axis=1)  # 将置信度升维
        # anchor_boxes=anchor_boxes.reshape(-1,8) #改变anchor_boxes的形状，与置信度拼接在一起
        print('----------------------anchor_boxes.shape:', anchor_boxes.shape)
        print('----------------------score.shape:', score.shape)
        total_boxes = np.hstack((anchor_boxes, score))  # 拼接在一起
    if pixel_boxes is not None and anchor_boxes is None:
        total_boxes=pixel_boxes
    if pixel_boxes is not None and anchor_boxes is not None:



        print('------------------------adjust_ratio之前的boxes.shape:',anchor_boxes.shape)
        print('---------------------------------scores.shape:',scores.shape)
        # new_anchor_boxes=np.expand_dims(anchor_boxes,axis=0)
        score=np.expand_dims(scores,axis=1) #将置信度升维

        # anchor_boxes=anchor_boxes.reshape(-1,8) #改变anchor_boxes的形状，与置信度拼接在一起
        print('----------------------anchor_boxes.shape:',anchor_boxes.shape)
        print('----------------------score.shape:',score.shape)

        anchor_total_boxes = np.hstack((anchor_boxes, score)) #拼接在一起
        anchor_total_boxes[:,8]+=0.5 #提升从anchor部分中出来的置信度设置为
        print('----------------------anchor_total_boxes.shape:', anchor_total_boxes.shape)
        total_boxes=np.vstack((anchor_total_boxes,pixel_boxes))
        print('------------------------total_boxes-------------------:',total_boxes.shape)

    if NMS_choice=='lanms': #使用lanms模块的nms
        total_boxes = lanms.merge_quadrangle_n9(total_boxes.astype('float32'), NMS_thresh)
        final_boxes = adjust_ratio(total_boxes, ratio_w, ratio_h)

        final_boxes[:,:8]/=img_size
        final_boxes[:,[0,2,4,6]]*=width
        final_boxes[:,[1,3,5,7]]*=heigth
        # print('-----------------------经过NMS的final_boxes:',final_boxes)
        # plot_img = plot_boxes(orignal_img, final_boxes)
        # plot_img.save(img_save_pths)
        return final_boxes

    if NMS_choice=='ssd': #使用SSD模块的nms
        score=total_boxes[:,8]
        total_boxes = total_boxes[:, :8]
        total_boxes=total_boxes.reshape(-1,4,2)
        keep = non_max_suppression_poly(total_boxes, score, NMS_thresh)  # anchor部分NMS
        total_boxes=total_boxes[keep]
        total_boxes=total_boxes.reshape(-1,8)
        total_boxes/=img_size
        total_boxes[:,[0,2,4,6]]*=width
        total_boxes[:,[1,3,5,7]]*=heigth

        final_boxes = adjust_ratio(total_boxes,ratio_w,ratio_h)
        return final_boxes
        # print('-----------------------经过NMS的final_boxes:', final_boxes)
        # plot_img=plot_boxes(orignal_img,final_boxes)
        # plot_img.save(img_save_pths)

def pixelandanchor_detect(img,model,device,img_save_path):#
    ' img PIL读取出来的图片'
    'model 加载权重后网络'
    'device 指定用几号gpu，还是cpu'

    img,ratio_h,ratio_w=resize_img(img)#   pixel中将图片转为可以整除32的长和边
    img_input=load_pil(img).to(device)
    #-------------------------pixel_detect部分----------------
    with torch.no_grad():
        score, geo,attetion_map, pre_location, pre_class = model(img_input)

    pixel_boxes=get_boxes(score.squeeze(0).cpu().numpy(),geo.squeeze(0).cpu().numpy(),if_eval=False)
    print('----------------------pixel_boxes.shape:',pixel_boxes.shape)
    pixel_boxes_finals=adjust_ratio(pixel_boxes,ratio_w,ratio_h)
    print('----------------------pixel_boxes_finals.shape:',pixel_boxes_finals.shape)
    plot_img = plot_boxes(img, pixel_boxes_finals)

    plot_img.save(img_save_path)

    #--------------------------anchor_detect部分---------------
    decoder=DataEncoder()

    print('---------------------pre_location.size():',pre_location.size())
    print('---------------------pre_class.size():',pre_class.size())
    print('------------------传入decode的img.size():',img.size)
    boxes, labels, scores =decoder.decode(pre_location.data.squeeze(0),pre_class.data.squeeze(0),img.size,if_eval=False)


    draw = ImageDraw.Draw(img)
    print('------------------------adjust_ratio之前的boxes.shape:',boxes.shape)

    boxes = adjust_ratio(boxes.reshape(-1,8),ratio_w,ratio_h)

    boxes = boxes.reshape(-1, 4, 2)
    print()

    for box in boxes:
        draw.polygon(np.expand_dims(box, 0), outline=(0, 255, 0))

    img.save(img_save_path)

from model.backbone import PixelAnchornet

if __name__ == '__main__':
    img_path='/data/yanghan/ICDAR2015/ch4_test_images/img_460.jpg'
    pth_path='/tmp/pycharm_project_389/newpths/model_epoch_400.pth'
    image_save_path='/tmp/pycharm_project_389/out_images'
    new_img_save_paths='/tmp/pycharm_project_389/out_images'+'{}'.format('/12_27_epoch400_new_test_460_ssd_1_size640.jpg')
    if not os.path.exists(image_save_path):
        os.mkdir(image_save_path)
    device=torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print('----------------------------------')
    net = PixelAnchornet(pretrained=False).to(device)
    model_checkpoint=torch.load(pth_path)
    net.load_state_dict(model_checkpoint)
    net.eval()
    img=Image.open(img_path)
    print('-----------------传入图片的img.size:',img.size)# 1280,720
    # pixelandanchor_detect(img,net,device,image_save_path+'{}'.format('/12_27_epoch400_test_460_selftwo.jpg'))
    pixle_anchor_detect(img,net,device,NMS_choice='ssd',img_save_pths=new_img_save_paths)













