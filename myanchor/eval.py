import torch
import shutil
import os
import time
from utils.anchorencoder import DataEncoder
import subprocess
from model.backbone import PixelAnchornet
from utils.detect_utils import detect_dataset,resize_img,load_pil,adjust_ratio
from PIL import Image


def anchor_detect(img, model, device,size_scale):
    '''detect text regions of img using model
    Input:
        img   : PIL Image
        model : detection model
        device: gpu if gpu is available
    Output:
        detected polys
    '''
    width,height=img.size[0],img.size[1]# 取得原图的宽和高
    img = img.resize((size_scale, size_scale)) #将推理图改为与训练图大小一致
    img, ratio_h, ratio_w = resize_img(img) # 改变输入图片的大小是图片的宽高均能被32整除
    with torch.no_grad():
        pre_location, pre_class = model(load_pil(img).to(device))  # 从网络出来的score值和geo值
    decoder = DataEncoder()


    boxes, labels, scores = decoder.decode(pre_location.data.squeeze(0), pre_class.data.squeeze(0), img.size)

    # print('---------------boxes.shape--------------:',boxes.shape)
    boxes=boxes.reshape(-1,8)
    boxes /= size_scale  # 除以所设置的图片size的的大小
    boxes[:,[0,2,4,6]]*=width
    boxes[:,[1,3,5,7]]*=height
    # boxes *= ([[width, height]] * 4)  # 还原为原图上的框
    boxes = adjust_ratio(boxes.reshape(-1, 8), ratio_w, ratio_h)
    return boxes#将最终的预测框



def detect_dataset(model, device, test_img_path, submit_path,size_scale=640):
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
        boxes = anchor_detect(Image.open(img_file), model, device,size_scale=size_scale)
        seq = []
        if boxes is not None:
            seq.extend([','.join([str(int(b)) for b in box[:]]) + '\n' for box in boxes])
        with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg', '.txt')), 'w') as f:
            f.writelines(seq)

def eval_model(model_pth,test_dir,submit_path,size_scale=640):
    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)
    os.mkdir(submit_path)

    device=("cuda:3" if torch.cuda.is_available() else "cpu")
    checkpoint=torch.load(model_pth)
    model = PixelAnchornet(pretrained=False).to(device)

    model.load_state_dict(checkpoint)
    start_time=time.time()
    detect_dataset(model,device,test_dir,submit_path,size_scale=size_scale)
    os.chdir(submit_path)
    subprocess.call('zip -q submit.zip *.txt')
    subprocess.call('mv submit.zip ../')
    os.chdir('../')
    subprocess('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
    os.remove('./submit.zip')
    print('eval time is {}'.format(time.time() - start_time))


if __name__ == '__main__':
    model_pth='/tmp/myanchor/focal_flat_loss_pths/model_epoch_25.pth'

    test_dir = '/data/yanghan/ICDAR2015/ch4_test_images'
    submit_pth='./submit_epoch25_size640'
    size_scale=640# 设置为与训练的时候resize后的大小一致，推理效果更好
    eval_model(model_pth,test_dir,submit_pth,size_scale=640)






