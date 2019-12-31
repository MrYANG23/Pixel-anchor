import math
import cv2
import numpy as np
from shapely.geometry import Polygon
from PIL import Image
import torch


def cal_distance(x1, y1, x2, y2):
    '''calculate the Euclidean distance'''
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def move_points(vertices, index1, index2, r, coef):
    '''move the two points to shrink edge
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        index1  : offset of point1
        index2  : offset of point2
        r       : [r1, r2, r3, r4] in paper
        coef    : shrink ratio in paper
    Output:
        vertices: vertices where one edge has been shinked
    '''
    index1 = index1 % 4  # 第几个点
    index2 = index2 % 4
    x1_index = index1 * 2 + 0  # 第几个点在vertices中的序号
    y1_index = index1 * 2 + 1
    x2_index = index2 * 2 + 0
    y2_index = index2 * 2 + 1

    r1 = r[index1]  # 第几个点中离该点最近的边。
    r2 = r[index2]
    length_x = vertices[x1_index] - vertices[x2_index]  # 两点间x轴间的距离
    length_y = vertices[y1_index] - vertices[y2_index]  # 两点间y轴间的距离
    length = cal_distance(vertices[x1_index], vertices[y1_index], vertices[x2_index], vertices[y2_index])  # 计算两点间的距离
    if length > 1:
        ratio = (r1 * coef) / length
        vertices[x1_index] += ratio * (-length_x)
        vertices[y1_index] += ratio * (-length_y)
        ratio = (r2 * coef) / length
        vertices[x2_index] += ratio * length_x
        vertices[y2_index] += ratio * length_y
    return vertices


def shrink_poly(vertices, coef=0.3):  # 返回收缩后的标签框坐标
    '''shrink the text region
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        coef    : shrink ratio in paper
    Output:
        v       : vertices of shrinked text region <numpy.ndarray, (8,)>
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    r1 = min(cal_distance(x1, y1, x2, y2), cal_distance(x1, y1, x4, y4))
    r2 = min(cal_distance(x2, y2, x1, y1), cal_distance(x2, y2, x3, y3))
    r3 = min(cal_distance(x3, y3, x2, y2), cal_distance(x3, y3, x4, y4))
    r4 = min(cal_distance(x4, y4, x1, y1), cal_distance(x4, y4, x3, y3))
    r = [r1, r2, r3, r4]

    # obtain offset to perform move_points() automatically
    if cal_distance(x1, y1, x2, y2) + cal_distance(x3, y3, x4, y4) > \
            cal_distance(x2, y2, x3, y3) + cal_distance(x1, y1, x4, y4):
        offset = 0  # two longer edges are (x1y1-x2y2) & (x3y3-x4y4)
    else:
        offset = 1  # two longer edges are (x2y2-x3y3) & (x4y4-x1y1)

    v = vertices.copy()
    v = move_points(v, 0 + offset, 1 + offset, r, coef)
    v = move_points(v, 2 + offset, 3 + offset, r, coef)
    v = move_points(v, 1 + offset, 2 + offset, r, coef)
    v = move_points(v, 3 + offset, 4 + offset, r, coef)
    return v


def get_rotate_mat(theta):  # 得到某点按照某弧度旋转后的所需要相乘的旋转矩阵
    '''positive theta value means rotate clockwise'''
    return np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])


def rotate_vertices(vertices, theta, anchor=None):  # 得到经过旋转后的坐标
    '''rotate vertices around anchor
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
        theta   : angle in radian measure#   弧度的意思
        anchor  : fixed position during rotation
    Output:
        rotated vertices <numpy.ndarray, (8,)>
    '''
    v = vertices.reshape((4, 2)).T  # 分为2*4的矩阵，每行分别代表每个点的x轴坐标，y轴坐标
    # print('传入的vertice,reshape((4,2))后再转置的shape：', v.shape)
    if anchor is None:
        anchor = v[:, :1]  # 在旋转过程中固定的点 (x1,y1)
        # print('vertice[:,:1].shape', anchor)
    rotate_mat = get_rotate_mat(theta)
    # print('rotate_mat:', rotate_mat, 'rotate_mat.shape:', rotate_mat.shape)
    # print('(v-anchor).shape:', (v - anchor).shape)
    res = np.dot(rotate_mat, v - anchor)  # v-anchor 每个点相对于固定点间的x轴和y轴距离
    # print('res.shape:', res.shape)
    # print('res+anchor.shape:', (res + anchor).shape)
    # print('(res+anchor).T.shape:', (res + anchor).T.shape)
    return (res + anchor).T.reshape(-1)


def get_boundary(vertices):  # 得到矩形定点集合中最小，最大的x,y值所构成的矩形顶点集合。
    '''get the tight boundary around given vertices
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the boundary
    '''
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    x_min = min(x1, x2, x3, x4)
    x_max = max(x1, x2, x3, x4)
    y_min = min(y1, y2, y3, y4)
    y_max = max(y1, y2, y3, y4)
    return x_min, x_max, y_min, y_max


def cal_error(vertices):  # 计算文本框和对应的矩形框间各个定点间欧拉距离间的和
    '''default orientation is x1y1 : left-top, x2y2 : right-top, x3y3 : right-bot, x4y4 : left-bot
    calculate the difference between the vertices orientation and default orientation
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        err     : difference measure
    '''
    x_min, x_max, y_min, y_max = get_boundary(vertices)
    x1, y1, x2, y2, x3, y3, x4, y4 = vertices
    err = cal_distance(x1, y1, x_min, y_min) + cal_distance(x2, y2, x_max, y_min) + \
          cal_distance(x3, y3, x_max, y_max) + cal_distance(x4, y4, x_min, y_max)
    return err


def find_min_rect_angle(vertices):  # 返回最佳的旋转角度
    '''find the best angle to rotate poly and obtain min rectangle
    Input:
        vertices: vertices of text region <numpy.ndarray, (8,)>
    Output:
        the best angle <radian measure>
    '''
    angle_interval = 1  # 角度间隔
    angle_list = list(range(-90, 90, angle_interval))
    # print('旋转的角度angle_list的列表:', angle_list)
    area_list = []
    for theta in angle_list:
        rotated = rotate_vertices(vertices, theta / 180 * math.pi)
        x1, y1, x2, y2, x3, y3, x4, y4 = rotated
        temp_area = (max(x1, x2, x3, x4) - min(x1, x2, x3, x4)) * \
                    (max(y1, y2, y3, y4) - min(y1, y2, y3, y4))
        area_list.append(temp_area)
    # print('通过计算文本框经过旋转不同角度后外接矩形的面积大小组合的列表area_list:', area_list)

    sorted_area_index = sorted(list(range(len(area_list))), key=lambda k: area_list[k])
    # print('排序后的sorted_area_index:', sorted_area_index)
    min_error = float('inf')
    best_index = -1
    rank_num = 10
    # find the best angle with correct orientation
    for index in sorted_area_index[:rank_num]:
        rotated = rotate_vertices(vertices, angle_list[index] / 180 * math.pi)
        temp_error = cal_error(rotated)
        if temp_error < min_error:
            min_error = temp_error
            best_index = index
    return angle_list[best_index] / 180 * math.pi


def is_cross_text(start_loc, length, vertices):
    '''check if the crop image crosses text regions
    Input:
        start_loc: left-top position
        length   : length of crop image
        vertices : vertices of text regions <numpy.ndarray, (n,8)>
    Output:
        True if crop image crosses text region
    '''
    if vertices.size == 0:
        return False
    start_w, start_h = start_loc
    a = np.array([start_w, start_h, start_w + length, start_h, \
                  start_w + length, start_h + length, start_w, start_h + length]).reshape((4, 2))
    p1 = Polygon(a).convex_hull
    for vertice in vertices:
        p2 = Polygon(vertice.reshape((4, 2))).convex_hull
        inter = p1.intersection(p2).area
        if 0.01 <= inter / p2.area <= 0.99:
            return True
    return False
def new_crop_img(img, vertices, labels, length):
	'''crop img patches to obtain batch and augment
	Input:
		img         : PIL Image
		vertices    : vertices of text regions <numpy.ndarray, (n,8)>
		labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
		length      : length of cropped image region
	Output:
		region      : cropped image region
		new_vertices: new vertices in cropped region
	'''

	h, w = img.height, img.width
	# confirm the shortest side of image >= length
	if h >= w and w < length:
		img = img.resize((length, int(h * length / w)), Image.BILINEAR)
	elif h < w and h < length:
		img = img.resize((int(w * length / h), length), Image.BILINEAR)
	ratio_w = img.width / w
	ratio_h = img.height / h
	assert(ratio_w >= 1 and ratio_h >= 1)

	new_vertices = np.zeros(vertices.shape)
	if vertices.size > 0:
		new_vertices[:,[0,2,4,6]] = vertices[:,[0,2,4,6]] * ratio_w
		new_vertices[:,[1,3,5,7]] = vertices[:,[1,3,5,7]] * ratio_h
	#find four limitate point by vertices
	vertice_x = [np.min(new_vertices[:, [0, 2, 4, 6]]), np.max(new_vertices[:, [0, 2, 4, 6]])]
	vertice_y = [np.min(new_vertices[:, [1, 3, 5, 7]]), np.max(new_vertices[:, [1, 3, 5, 7]])]
	# find random position
	remain_w = [0,img.width - length]
	remain_h = [0,img.height - length]
	if vertice_x[1]>length:
		remain_w[0] = vertice_x[1] - length
	if vertice_x[0]<remain_w[1]:
		remain_w[1] = vertice_x[0]
	if vertice_y[1]>length:
		remain_h[0] = vertice_y[1] - length
	if vertice_y[0]<remain_h[1]:
		remain_h[1] = vertice_y[0]

	start_w = int(np.random.rand() * (remain_w[1]-remain_w[0]))+remain_w[0]
	start_h = int(np.random.rand() * (remain_h[1]-remain_h[0]))+remain_h[0]
	box = (start_w, start_h, start_w + length, start_h + length)
	region = img.crop(box)
	if new_vertices.size == 0:
		return region, new_vertices

	new_vertices[:,[0,2,4,6]] -= start_w
	new_vertices[:,[1,3,5,7]] -= start_h


	return region, new_vertices


def crop_img(img, vertices, labels, length):
    '''crop img patches to obtain batch and augment
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        labels      : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        length      : length of cropped image region
    Output:
        region      : cropped image region
        new_vertices: new vertices in cropped region
    '''
    h, w = img.height, img.width
    # confirm the shortest side of image >= length
    if h >= w and w < length:
        img = img.resize((length, int(h * length / w)), Image.BILINEAR)
    elif h < w and h < length:
        img = img.resize((int(w * length / h), length), Image.BILINEAR)
    ratio_w = img.width / w
    ratio_h = img.height / h
    assert (ratio_w >= 1 and ratio_h >= 1)

    new_vertices = np.zeros(vertices.shape)
    if vertices.size > 0:
        new_vertices[:, [0, 2, 4, 6]] = vertices[:, [0, 2, 4, 6]] * ratio_w
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * ratio_h

    # find random position
    remain_h = img.height - length
    remain_w = img.width - length
    flag = True
    cnt = 0
    while flag and cnt < 1000:
        cnt += 1
        start_w = int(np.random.rand() * remain_w)
        start_h = int(np.random.rand() * remain_h)
        flag = is_cross_text([start_w, start_h], length, new_vertices[labels == 1, :])
    box = (start_w, start_h, start_w + length, start_h + length)
    region = img.crop(box)
    if new_vertices.size == 0:
        return region, new_vertices

    new_vertices[:, [0, 2, 4, 6]] -= start_w
    new_vertices[:, [1, 3, 5, 7]] -= start_h
    return region, new_vertices


def rotate_all_pixels(rotate_mat, anchor_x, anchor_y, length):
    '''get rotated locations of all pixels for next stages
    Input:
        rotate_mat: rotatation matrix
        anchor_x  : fixed x position
        anchor_y  : fixed y position
        length    : length of image
    Output:
        rotated_x : rotated x positions <numpy.ndarray, (length,length)>
        rotated_y : rotated y positions <numpy.ndarray, (length,length)>
    '''
    x = np.arange(length)# numpy.ndarray (length)
    y = np.arange(length)# numpy.ndarray (length)
    x, y = np.meshgrid(x, y)
    #x numpy.ndarray(length , length)
    #y numpy.ndarray(length , length)
    x_lin = x.reshape((1, x.size))
    # #x numpy.ndarray(1,length * length)
    y_lin = y.reshape((1, x.size)) # 1*n
    # y numpy.ndarray(1,length * length)


    coord_mat = np.concatenate((x_lin, y_lin), 0)# 得到每个像素点的坐标 （2，n）n为像素点的个数
    # coord_mat numpy.ndarray(2,length*length)
    rotated_coord = np.dot(rotate_mat, coord_mat - np.array([[anchor_x], [anchor_y]])) + \
                    np.array([[anchor_x], [anchor_y]])

    # print('rotated_coord:', rotated_coord.shape)
    rotated_x = rotated_coord[0, :].reshape(x.shape)# ndarray(length,length) 旋转后得到所有像素点的x方向上坐标
    rotated_y = rotated_coord[1, :].reshape(y.shape)# ndarray(length,length) 旋转后得到所有像素点上y方向上的坐标
    return rotated_x, rotated_y


def adjust_height(img, vertices, ratio=0.2):
    '''adjust height of image to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        ratio       : height changes in [0.8, 1.2]
    Output:
        img         : adjusted PIL Image
        new_vertices: adjusted vertices
    '''
    ratio_h = 1 + ratio * (np.random.rand() * 2 - 1)
    old_h = img.height
    new_h = int(np.around(old_h * ratio_h))
    img = img.resize((img.width, new_h), Image.BILINEAR)

    new_vertices = vertices.copy()
    if vertices.size > 0:
        new_vertices[:, [1, 3, 5, 7]] = vertices[:, [1, 3, 5, 7]] * (new_h / old_h)
    return img, new_vertices


def rotate_img(img, vertices, angle_range=10):
    '''rotate image [-10, 10] degree to aug data
    Input:
        img         : PIL Image
        vertices    : vertices of text regions <numpy.ndarray, (n,8)>
        angle_range : rotate range
    Output:
        img         : rotated PIL Image
        new_vertices: rotated vertices
    '''
    center_x = (img.width - 1) / 2
    center_y = (img.height - 1) / 2
    angle = angle_range * (np.random.rand() * 2 - 1)
    img = img.rotate(angle, Image.BILINEAR)
    new_vertices = np.zeros(vertices.shape)
    for i, vertice in enumerate(vertices):
        new_vertices[i, :] = rotate_vertices(vertice, -angle / 180 * math.pi, np.array([[center_x], [center_y]]))
    return img, new_vertices


def get_score_geo(img, vertices, labels, scale, length):  # 制作score的标签以及geometry的标签。
    '''generate score gt and geometry gt
    Input:
        img     : PIL Image
        vertices: vertices of text regions <numpy.ndarray, (n,8)>
        labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
        scale   : feature map / image
        length  : image length
    Output:
        score gt, geo gt, ignored
    '''
    # print('----------------------------img.height:',img.height)
    # print('----------------------------img.width:',img.width)
    score_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)  # 特征图上score的大小
    geo_map = np.zeros((int(img.height * scale), int(img.width * scale), 5), np.float32)  # 特征图geo的标签大小
    # print('-------------geo_map.size():',geo_map.shape)
    ignored_map = np.zeros((int(img.height * scale), int(img.width * scale), 1), np.float32)  # 特征图上的mask

    index = np.arange(0, length, int(1 / scale))
    index_x, index_y = np.meshgrid(index, index)
    # print('index_x.shaape:', index_x.shape, 'index_y.shape:', index_y.shape)
    ignored_polys = []
    polys = []

    # print('enumrate中vertices的type:', type(vertices))
    # print('enumrate中vertices.shape:', vertices.shape, 'vertices:', vertices)

    for i, vertice in enumerate(vertices):
        # print('----------------------------------每次做shrink_poly前的vertice---------:', vertice)
        if labels[i] == 0:
            ignored_polys.append(np.around(scale * vertice.reshape((4, 2))).astype(np.int32))
            continue

        poly = np.around(scale * shrink_poly(vertice).reshape((4, 2))).astype(np.int32)  # scaled & shrinked
        # print('poly.shape:', poly.shape)
        # print('-----------------------------------每次经过shrink_poly后的vertice--------：', poly)
        # 返回乘以缩放因子后的收缩多边形的顶点集合。
        polys.append(poly)
        temp_mask = np.zeros(score_map.shape[:-1], np.float32)
        # print('使用fillpoly前的temp_mask.shape:', temp_mask.shape)
        cv2.fillPoly(temp_mask, [poly], 1)  # 作为geo部分的掩码，对应于经过缩放后的多边形文本框
        # print('使用fillpoly后的tem_mask.shape:', temp_mask.shape)

        theta = find_min_rect_angle(vertice)  # 得到最佳旋转角度
        rotate_mat = get_rotate_mat(theta)  # 得到最佳旋转角度的旋转矩阵

        rotated_vertices = rotate_vertices(vertice, theta)  # 得到按照最佳旋转角度旋转后的文本框坐标
        x_min, x_max, y_min, y_max = get_boundary(rotated_vertices)  # 得到需要计算像素点到矩形框四边距离的矩形框。
        rotated_x, rotated_y = rotate_all_pixels(rotate_mat, vertice[0], vertice[1], length)
        # ndarray(length,length) 旋转后得到所有像素点的x方向上坐标
        # ndarray(length,length) 旋转后得到所有像素点的y方向上坐标

        d1 = rotated_y - y_min
        # print('ex_d1.shaep:', d1.shape)
        d1[d1 < 0] = 0
        # print('d1.shape:', d1.shape)
        d2 = y_max - rotated_y
        # print('ex_d2.shape:', d2.shape)
        d2[d2 < 0] = 0
        # print('d2.shape:', d2.shape)
        d3 = rotated_x - x_min
        # print('ex_d3.shape:', d3.shape)
        d3[d3 < 0] = 0
        # print('d3.shape:', d3.shape)
        d4 = x_max - rotated_x
        # print('ex_d4.shape:', d4.shape)
        d4[d4 < 0] = 0
        # print('ex_d4.shape:', d4.shape)

        # print('index_x:', index_x, 'index_y:', index_y)
        # print('d1[index_y, index_x]:', d1[index_y, index_x].shape)
        geo_map[:, :, 0] += d1[index_y, index_x] * temp_mask
        geo_map[:, :, 1] += d2[index_y, index_x] * temp_mask
        geo_map[:, :, 2] += d3[index_y, index_x] * temp_mask
        geo_map[:, :, 3] += d4[index_y, index_x] * temp_mask
        geo_map[:, :, 4] += theta * temp_mask
    # print('------------------------------------------需要在score_map上哪些位置填1的polys的集合-----------------:', polys)
    cv2.fillPoly(ignored_map, ignored_polys, 1)
    cv2.fillPoly(score_map, polys, 1)  # 得到每张图上的score_map，fillpoly以score_map为背景，polys为对应的标签。
    # print('-------------------------------------从dataset中传出的score_map（gt_score）:',score_map,'score_map.size():',score_map.shape)

    return torch.Tensor(score_map).permute(2, 0, 1), torch.Tensor(geo_map).permute(2, 0, 1), torch.Tensor(
        ignored_map).permute(2, 0, 1)

def extract_vertices(lines):
    '''extract vertices info from txt lines
    Input:
        lines   : list of string info
    Output:
        vertices: vertices of text regions <numpy.ndarray, (n,8)>
        labels  : 1->valid, 0->ignore, <numpy.ndarray, (n,)>
    '''
    labels = []
    vertices = []
    for line in lines:
        vertices.append(list(map(int, line.rstrip('\n').lstrip('\ufeff').split(',')[:8])))
        label = 0 if '###' in line else 1  # 标签中有###视为不合格的标签。
        labels.append(label)
    return np.array(vertices), np.array(labels)

