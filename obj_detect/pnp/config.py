from cv2 import cv2 as cv


#小地图
class little_map:
    #地图宽（横的）
    __map_width=600
    #地图高（竖的）
    __map_height=600
    '''
    python中双下划线开头表示私有属性
    '''
    def __init__(self):
        self.pic=cv.imread('/home/truth/github/TJRM21/radar/locate_and_mapping/court.jpeg')
        cv.imshow("court",self.pic)
        size=(self.__map_width,self.__map_height)
        self.pic=cv.resize(self.pic,size)
    def get_width(self):
        return self.__map_width
    def get_height(self):
        return self.__map_height

lm=little_map()