from cv2 import cv2 as cv
from pnp.config import lm

def show_little_map(points_2d):
    '''
    显示小地图,并在地图上标记目标
    :param points_2d: 待标记的二维点坐标
    :return: Nonegi
    '''
    for point in points_2d:
        cv.circle(lm.pic,point,10,(0,255,0),-1)
    cv.imshow('小地图',lm.pic)

def transform_3dpoints_to_2d(points_3d):
    '''
    影响因素多，待定

    将三维点转换成可显示在小地图上的二维点
    :param points_3d: 待转换的三维点坐标
    :return: points_2d
    '''
    points_2d=[]
    for point in points_3d:
        
        points_2d.append(point)
    return points_2d

# 测试用
# if __name__ == "__main__":
#     t=((1,20),(100,300))
#     show_little_map(t)
#     cv.waitKey()