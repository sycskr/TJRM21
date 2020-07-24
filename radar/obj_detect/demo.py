# coding:utf-8
'''
此文件实现地面机器人的目标检测
后续优化 ：
    1.分类只选取地面机器人的车身，然后对车身内部进行常规装甲板匹配（权重文件重新训练
    2.加入测距与放射变换
    3.DEEP_SORT 内部需要特征提取的分类器
    修改日期：07.22 truth 初步完成车身识别 , 加入了DEEP_SORT，自带卡尔曼滤波
'''

import argparse
import torch.backends.cudnn as cudnn

import numpy as np
import cv2 as cv

from yolov5.models.experimental import *
from yolov5.rotate_bound import rotate_bound
from utils.datasets import *
from utils.utils import *
from utils.draw import *
from utils.parser import *
from yolov5.models.experimental import attempt_load
from yolov5.rotate_bound import *
from deep_sort import *

def adjust_img(im0s,imgsz,device):
    '''
    调整图像的属性
    :param im0s: the original input by cv.imread
           imgsz: the size
    :return: img for input
    '''
    # Padded resize
    img = letterbox(im0s, new_shape=imgsz)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    #转成tensor
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    return img

def detect_per_frame(im0s):
    '''

    :param im0s:
    :return:bbox_xywh, cls_conf, cls_ids
    '''
    '''
    need two images 
        @ img  is the adjusted image as the input of the DNN
        @ im0s is the orignial image
    '''
    img = adjust_img(im0s, imgsz, device)
    # inference 推断

    pred = models(img)[0]
    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic)


    bbox_xcycwh = []
    cls_conf  = []
    cls_ids   = []

    # Process detections
    for i, det in enumerate(pred):  # detections per image
        '''
        pred is a tensor list which as six dim
            @dim 0-3 : upper-left (x1,y1) to right-bottom (x2,y2) 就是我们需要的矩形框
            @dim 4 confidence 
            @dim 5 class_index 类名
        '''
        # gn = torch.tensor(im0s.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            # 选择前四项，作为缩放依据
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()
            cls_conf = det[:, 4]
            cls_ids  = det[:, 5]

            #print(det)

            # Print results
            # output = ''
            # for c in det[:, -1].unique():
            #     n = (det[:, -1] == c).sum()  # detections per class
            #     output += '%g %ss, ' % (n, names[int(c)])  # add to string

            # # Draw rectangles
            for *xyxy, conf, cls in det:
                # print("xyxy : ", xyxy, "\n conf : ", conf, "cls : ", cls)
                #label = '%s %.2f' % (names[int(cls)], conf)
                #plot_one_box(xyxy, im0s, label=label, color=colors[int(cls)], line_thickness=3)
                xywh=[(xyxy[0]+xyxy[2])/2, (xyxy[1]+xyxy[3])/2, xyxy[2]-xyxy[0], xyxy[3]-xyxy[1]]
                bbox_xcycwh.append(xywh)

            # Print time (inference + NMS)

            #cv2.imshow("test", im0s)
            #cv.waitKey(1)

    return bbox_xcycwh, cls_conf, cls_ids

#GPU设备
device_ = ''
#权重
weights = 'yolov5/weights/last_yolov5s_0722.pt'
#输入文件目录
source = 'yolov5/inference/images'  # file/folder, 0 for webcam
#输出文件目录
out = 'inference/output'  # output folder
#固定输入大小？
imgsz = 640  # help='inference size (pixels)')
#置信度阈值
conf_thres = 0.4
#iou合并阈值
iou_thres = 0.5
#deep_sort configs
deep_sort_configs='configs/deep_sort.yaml'

classes = ''
agnostic = ''

# Initialize 找GPU
device = torch_utils.select_device(device_)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model载入模型
models = attempt_load(weights, map_location=device)  # load FP32 model
imgsz = check_img_size(imgsz, s=models.stride.max())  # check img_size
if half:
    models.half()  # to FP16

# Get names and colors获得类名与颜色
names = models.module.names if hasattr(models, 'module') else models.names
colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]
cfg = get_config(deep_sort_configs)



# results = []
# frame = cv.imread('/home/truth/ClionProjects/mySITP/yolo/yolov5/inference/images/bus.jpg')
#
# bbox_xcycwh, cls_conf, cls_ids = detect_per_frame(frame)
# print("xywh : \n" , bbox_xywh[0])
# print("confidence : \n",cls_conf)
# print("id : \n", cls_ids)
#

my_deepsort = build_tracker(cfg, torch.cuda.is_available())



cap = cv.VideoCapture("/home/truth/ClionProjects/mySITP/yolo/yolov5/inference/t1.mp4")  # 打开指定路径上的视频文件
iii=0
while (True):
    iii=iii+1
    if iii%3!=1 :
        continue
    ret, frame = cap.read()# BGR
    if ret == True:
        #rotate my video 因为视频是歪的= =
        height, width = frame.shape[:2]
        im0s = rotate_bound(frame, -90)
        #cv.imshow("video", im0s)

        t1 = torch_utils.time_synchronized()

        bbox_xcycwh, cls_conf, cls_ids = detect_per_frame(im0s)
        outputs = my_deepsort.update(bbox_xcycwh, cls_conf, im0s)

        t2 = torch_utils.time_synchronized()
        print('%s is detected. (%.3fs)' % (len(outputs), t2 - t1))

        # draw boxes for visualization
        if len(outputs) > 0:
            bbox_tlwh = []
            bbox_xyxy = outputs[:, :4]
            identities = outputs[:, -1]
            im0s = draw_boxes(im0s, bbox_xyxy, identities)

            cv2.imshow("test", im0s)
            cv.waitKey(1)

            # for bb_xyxy in bbox_xyxy:
            #     bbox_tlwh.append(DeepSort._xyxy_to_tlwh(bb_xyxy))
            #
            # results.append((1, bbox_tlwh, identities))

    else:
        break

cap.release()
cv.destroyAllWindows()
