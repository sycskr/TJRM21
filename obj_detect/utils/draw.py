import numpy as np
import cv2

palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)


def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_boxes(img, bbox,angles,distance,tvec, identities=None, offset=(0,0)):
    '''

    Parameters
    ----------
    img  :原图
    bbox :目标框
    angles:偏转角度
    distance:距离
    tvec:三维坐标
    identities:编号
    offset:偏移量
    Returns
    -------
    '''
    for i,box in enumerate(bbox):
        x1,y1,x2,y2 = [int(i) for i in box]
        angle =angles[i]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        #选目标的颜色
        color = compute_color_for_labels(id)
        label = '{}{:d}'.format("", id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        cv2.rectangle(img,(x1, y1),(x2,y2),color,3)
        cv2.rectangle(img,(x1, y1),(x1+t_size[0]+3,y1+t_size[1]+4), color,-1)
        cv2.putText(img,label,(x1,y1+t_size[1]+4), cv2.FONT_HERSHEY_PLAIN, 2, [255,255,255], 2)
        #打印相关信息
        HA =  '{}{:.2f}'.format("HA : ", angle[0]/3.14*180)
        cv2.putText(img, HA
                    , (x2, y2 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 2)

        VA =  '{}{:.2f}'.format("VA : ",-angle[1]/3.14*180)
        cv2.putText(img, VA, (x2, y2 + t_size[1]*2 + 8), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 2)

        Dist =  '{}{:.2f}'.format("Dist : ",distance[i]/1000)
        cv2.putText(img, Dist, (x2, y2 + t_size[1]*3 + 12), cv2.FONT_HERSHEY_PLAIN, 1, [0, 255, 0], 2)

    return img



if __name__ == '__main__':
    for i in range(82):
        print(compute_color_for_labels(i))
