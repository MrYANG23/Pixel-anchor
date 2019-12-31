import lanms
import numpy as np
import torch
from torchvision import transforms
from PIL import Image,ImageDraw
from utils.pixelutils import get_rotate_mat
import os

#------------------------------------pixel部分的detect_utils--------------
def resize_img(img):
    '''resize image to be divisible by 32
    '''
    w, h = img.size
    resize_w = w
    resize_h = h

    resize_h = resize_h if resize_h % 32 == 0 else int(resize_h / 32) * 32
    resize_w = resize_w if resize_w % 32 == 0 else int(resize_w / 32) * 32
    img = img.resize((resize_w, resize_h), Image.BILINEAR)
    ratio_h = resize_h / h
    ratio_w = resize_w / w
    print('--------------------------pixel中resize img后的img.size:',img.size)
    return img, ratio_h, ratio_w

def load_pil(img):
    '''convert PIL Image to torch.Tensor
    '''
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return t(img).unsqueeze(0)

def is_valid_poly(res, score_shape, scale):
    '''check if the poly in image scope
    Input:
        res        : restored poly in original image
        score_shape: score map shape
        scale      : feature map -> image
    Output:
        True if valid
    '''
    cnt = 0
    for i in range(res.shape[1]):
        if res[0, i] < 0 or res[0, i] >= score_shape[1] * scale or \
                res[1, i] < 0 or res[1, i] >= score_shape[0] * scale:
            cnt += 1
    return True if cnt <= 1 else False

def restore_polys(valid_pos, valid_geo, score_shape, scale=4):
    '''restore polys from feature maps in given positions
    Input:
        valid_pos  : potential text positions <numpy.ndarray, (n,2)>
        valid_geo  : geometry in valid_pos <numpy.ndarray, (5,n)>
        score_shape: shape of score map
        scale      : image / feature map
    Output:
        restored polys <numpy.ndarray, (n,8)>, index
    '''
    polys = []  # 返回的最终的预测框
    index = []  # 返回的最终预测框的序列id
    valid_pos *= scale  # 得到原图上有效点的坐标
    d = valid_geo[:4, :]  # 4 x N   #得到有效的每个像素点与预测框之间的距离
    angle = valid_geo[4, :]  # N,   #得到每个像素点的预测框角度
    for i in range(valid_pos.shape[0]):  # 循环每个有效的像素点
        x = valid_pos[i, 0]  # 每个有效像素点的x坐标
        y = valid_pos[i, 1]  # --------------y坐标
        y_min = y - d[0, i]  # 像素点对应预测框 ymin
        y_max = y + d[1, i]  # -------------- ymax
        x_min = x - d[2, i]  # -------------- xmin
        x_max = x + d[3, i]  # -------------- xmax
        rotate_mat = get_rotate_mat(-angle[i])  # 每个像素点对应的旋转角度矩阵

        temp_x = np.array([[x_min, x_max, x_max, x_min]]) - x  # 每个像素点对应的预测框4个点x的坐标与像素点坐标x间的距离
        temp_y = np.array([[y_min, y_min, y_max, y_max]]) - y  # 每个像素点对应的预测框4个点y的坐标与像素点坐标y间的距离
        coordidates = np.concatenate((temp_x, temp_y), axis=0)  # 2*4
        res = np.dot(rotate_mat, coordidates)  # 得到旋转后的偏差值
        res[0, :] += x
        res[1, :] += y
        # 最后得到的res为旋转后的预测框坐标

        if is_valid_poly(res, score_shape, scale):  # 判断是否为有效的框
            index.append(i)
            polys.append([res[0, 0], res[1, 0], res[0, 1], res[1, 1], res[0, 2], res[1, 2], res[0, 3], res[1, 3]])
    return np.array(polys), index

def get_boxes(score, geo, score_thresh=0.9, nms_thresh=0.2,if_eval=True):
    '''get boxes from feature map
    Input:
        score       : score map from model <numpy.ndarray, (1,row,col)>
        geo         : geo map from model <numpy.ndarray, (5,row,col)>
        score_thresh: threshold to segment score map
        nms_thresh  : threshold in nms
    Output:
        boxes       : final polys <numpy.ndarray, (n,9)>
    '''
    score = score[0, :, :]  # 去掉通道1的维度
    print('--------------------------二维的score.shape-------------------:',score.shape)
    xy_text = np.argwhere(score > score_thresh)  # n x 2, format is [r, c] #
    print('-------------------------xy_text.shape:',xy_text.shape)
    # 得到score大于置信度阈值的坐标 (n,2),n为n个像素点
    if xy_text.size == 0:
        return None
    xy_text = xy_text[np.argsort(xy_text[:, 0])]  # 将置信度阈值大于一定值的坐标，
    # 按照每个点的行坐标进行排序后的得到的xy_text中的从小到大的行索引，
    valid_pos = xy_text[:, ::-1].copy()  # n x 2, [x, y]# 将y,x 转换为x,y
    print('------------------valid_pos.shape()--------------:',valid_pos.shape)



    # 有效的坐标点
    valid_geo = geo[:, xy_text[:, 0], xy_text[:, 1]]  # 5 x n #经过阈值筛选后的有效geo 5*n n为有效像素点的个数。
    print('-----------------------------valid_geo.shape---------------------:',valid_geo.shape)

    polys_restored, index = restore_polys(valid_pos, valid_geo, score.shape)  # 得到最终的预测框集合，以及在valid_pos中的id序号


    if polys_restored.size == 0:
        return None
    boxes = np.zeros((polys_restored.shape[0], 9), dtype=np.float32)  #
    boxes[:, :8] = polys_restored  # 装最终所有预测框4个点的信息
    boxes[:, 8] = score[xy_text[index, 0], xy_text[index, 1]]  # 对应预测框的置信度值
    if if_eval:
        return boxes
    boxes = lanms.merge_quadrangle_n9(boxes.astype('float32'), nms_thresh)  #  经过NMS返回最终的预测框

    print('-----------------pixel中未经过NMS后的boxes.shape---------------:',boxes.shape)
    return boxes

def adjust_ratio(boxes, ratio_w, ratio_h):
    '''refine boxes
    Input:
        boxes  : detected polys <numpy.ndarray, (n,9)>
        ratio_w: ratio of width
        ratio_h: ratio of height
    Output:
        refined boxes
    '''
    if boxes is None or boxes.size == 0:
        return None
    boxes[:, [0, 2, 4, 6]] /= ratio_w
    boxes[:, [1, 3, 5, 7]] /= ratio_h
    return np.around(boxes)

def detect(img, model, device):
    '''detect text regions of img using model
    Input:
        img   : PIL Image
        model : detection model
        device: gpu if gpu is available
    Output:
        detected polys
    '''
    img, ratio_h, ratio_w = resize_img(img)
    with torch.no_grad():
        score, geo = model(load_pil(img).to(device))  # 从网络出来的score值和geo值
    boxes = get_boxes(score.squeeze(0).cpu().numpy(), geo.squeeze(0).cpu().numpy())
    return adjust_ratio(boxes, ratio_w, ratio_h) #将最终的预测框

def plot_boxes(img, boxes):
    '''plot boxes on image
    '''
    if boxes is None:
        print('boxes is none')
        return img

    draw = ImageDraw.Draw(img)
    for box in boxes:

        draw.polygon([box[0], box[1], box[2], box[3], box[4], box[5], box[6], box[7]], outline=(0, 0, 255))
    return img

def detect_dataset(model, device, test_img_path, submit_path):
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
        boxes = detect(Image.open(img_file), model, device)
        seq = []
        if boxes is not None:
            seq.extend([','.join([str(int(b)) for b in box[:-1]]) + '\n' for box in boxes])
        with open(os.path.join(submit_path, 'res_' + os.path.basename(img_file).replace('.jpg', '.txt')), 'w') as f:
            f.writelines(seq)


#----------------------------------anchor部分detect_utils-------------------------------




