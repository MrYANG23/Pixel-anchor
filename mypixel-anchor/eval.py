import torch
import time
import os
import subprocess
import shutil
from model.backbone import PixelAnchornet
from utils.detect_utils import detect_dataset,resize_img,load_pil,adjust_ratio
from PIL import Image
from utils.detect_utils import plot_boxes,load_pil,get_boxes,resize_img,detect,adjust_ratio
from utils.anchorencoder import DataEncoder
import numpy as np
import lanms
from pixel_anchor_detect import pixle_anchor_detect

#改变了推理图的大小设置为与训练的时候一致
# def pixle_anchor_detect(img,model,device,NMS_choice='',NMS_thresh=0.2,img_size=640):
#
#     width,heigth=img.size[0],img.size[1]
#     original_img=img
#     img.resize((img_size,img_size))#改变推理过程中的输入图大小，设置与训练图大小一致。
#
#     img,ratio_h,ratio_w=resize_img(img)#   pixel中将图片转为可以整除32的长和边
#     img_input=load_pil(img).to(device)
#     #-------------------------pixel_detect部分----------------
#     with torch.no_grad():
#         score, geo, attention_map,pre_location, pre_class = model(img_input)
#         print('-----------------------从网络输出的score.shape:',score.shape)
#     pixel_boxes=get_boxes(score.squeeze(0).cpu().numpy(),geo.squeeze(0).cpu().numpy())
#     print('----------------------pixel_boxes.shape:',pixel_boxes.shape)
#     # pixel_boxes_finals=adjust_ratio(pixel_boxes,ratio_w,ratio_h)
#     # print('----------------------pixel_boxes_finals:',pixel_boxes_finals.shape)
#     # plot_img = plot_boxes(img, pixel_boxes_finals)
#     #--------------------------anchor_detect部分---------------
#     decoder=DataEncoder()
#     # print('---------------------pre_location.size():',pre_location.size())
#     # print('---------------------pre_class.size():',pre_class.size())
#     # print('------------------传入decode的img.size():',img.size)
#     anchor_boxes, labels, scores =decoder.decode(pre_location.data.squeeze(0),pre_class.data.squeeze(0),img.size,if_eval=True)
#
#     # print('--------------------decode出来后的anchor_boxe.shape:',anchor_boxes.shape)
#     # print('--------------------decode出来后的labels.shape:',labels.shape)
#     # print('--------------------decode出来后的scores.shape:',scores.shape)
#     #
#     # print('------------------------adjust_ratio之前的boxes.shape:',anchor_boxes.shape)
#     # new_anchor_boxes=np.expand_dims(anchor_boxes,axis=0)
#     scores=np.expand_dims(scores,axis=1)
#     # print('----------------scores.shape-------------:',scores.shape)
#     anchor_total_boxes = np.hstack((anchor_boxes, scores))
#     # print('----------------------anchor_total_boxes.shape:', anchor_total_boxes.shape)
#     total_boxes=np.vstack((anchor_total_boxes,pixel_boxes))
#     # print('------------------------total_boxes-------------------:',total_boxes.shape)
#     if NMS_choice=='lanms':
#         final_boxes = lanms.merge_quadrangle_n9(total_boxes.astype('float32'), NMS_thresh)
#
#         #还原到原图上
#         final_boxes[:,0:8]/=img_size
#         final_boxes[:,[0,2,4,6]]*=width
#         final_boxes[:,[1,3,5,7]]*=heigth
#         final_boxes = adjust_ratio(final_boxes,ratio_w,ratio_h)
#     if NMS_choice=='ssd': #使用SSD模块的nms
#         score=total_boxes[:,8]
#         total_boxes = total_boxes[:, :8]
#         total_boxes=total_boxes.reshape(-1,4,2)
#         keep = non_max_suppression_poly(total_boxes, score, NMS_thresh)  # anchor部分NMS
#         total_boxes=total_boxes[keep]
#         total_boxes=total_boxes.reshape(-1,8)
#         total_boxes/=img_size
#         total_boxes[:,[0,2,4,6]]*=width
#         total_boxes[:,[1,3,5,7]]*=heigth
#
#         final_boxes = adjust_ratio(total_boxes,ratio_w,ratio_h)
#         print('-----------------------经过NMS的final_boxes:', final_boxes)
#         plot_img=plot_boxes(orignal_img,final_boxes)
#         plot_img.save(img_save_pths)

def detect_dataset(model, device, test_img_path, submit_path,NMS_choice=''):
    '''detection on whole dataset, save .txt results in submit_path
    Input:
        model        : detection model
        device       : gpu if gpu is available
        test_img_path: dataset path
        submit_path  : submit result for evaluation
    '''
    img_files = os.listdir(test_img_path)
    img_files = sorted([os.path.join(test_img_path, img_file) for img_file in img_files])

    for i, img_file in enumerate(img_files):
        print('evaluating {} image'.format(i), end='\r')
        boxes = pixle_anchor_detect(Image.open(img_file), model, device,NMS_choice=NMS_choice)
        seq = []
        if boxes is not None:
            seq.extend([','.join([str(int(b)) for b in box[:]]) + '\n' for box in boxes])
        with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg', '.txt')), 'w') as f:
            f.writelines(seq)

def eval_model(model_pth,test_dir,submit_path,NMS_choice='lanms'):
    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)
    os.mkdir(submit_path)

    device=("cuda:2" if torch.cuda.is_available() else "cpu")
    checkpoint=torch.load(model_pth)
    model = PixelAnchornet(pretrained=False).to(device)

    model.load_state_dict(checkpoint)
    start_time=time.time()
    detect_dataset(model,device,test_dir,submit_path,NMS_choice=NMS_choice)
    os.chdir(submit_path)
    subprocess.call('zip -q submit.zip *.txt')
    subprocess.call('mv submit.zip ../')
    os.chdir('../')
    subprocess('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
    os.remove('./submit.zip')
    print('eval time is {}'.format(time.time() - start_time))


if __name__ == '__main__':
    model_pth = '/tmp/pycharm_project_389/newpths/model_epoch_400.pth'

    test_dir = '/data/yanghan/ICDAR2015/ch4_test_images'
    submit_pth = './submit_epoch400'
    eval_model(model_pth, test_dir, submit_pth,NMS_choice='lanms')