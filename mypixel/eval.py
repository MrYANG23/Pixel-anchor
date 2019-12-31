import torch
from utils.detect_utils import detect_dataset,detect

from model.backbone import PixelAnchornet
import subprocess
import os
import time
import shutil

def eval_model(model_pth,test_dir,submit_path,img_size=640):
    if os.path.exists(submit_path):
        shutil.rmtree(submit_path)
    os.mkdir(submit_path)

    device=("cuda:3" if torch.cuda.is_available() else "cpu")
    checkpoint=torch.load(model_pth)
    model = PixelAnchornet(pretrained=False).to(device)

    model.load_state_dict(checkpoint)
    start_time=time.time()
    detect_dataset(model,device,test_dir,submit_path,img_size=640)
    os.chdir(submit_path)
    res = subprocess.getoutput('zip -q submit.zip *.txt')
    res = subprocess.getoutput('mv submit.zip ../')
    os.chdir('../')
    res = subprocess.getoutput('python ./evaluate/script.py –g=./evaluate/gt.zip –s=./submit.zip')
    print('--------------------res-----------------------',res)
    os.remove('./submit.zip')
    print('eval time is {}'.format(time.time() - start_time))


if __name__ == '__main__':
    model_pth='/tmp/mypixel/pths/model_epoch_400.pth'
    test_dir='/data/yanghan/ICDAR2015/ch4_test_images'
    submit_pth='./submit_inputsize640'
    eval_model(model_pth,test_dir,submit_pth)







