from cv2 import cv2 as cv


#小地图
class little_map:

    #地图宽（横的）
    map_width=1920
    #地图高（竖的）
    map_height=886

    #地图宽（横的）
    count_width=25000
    #地图高（竖的）
    count_height=15000

    '''
    python中双下划线开头表示私有属性
    '''
    def __init__(self):
        self.pic=cv.imread('/home/truth/github/TJRM21/radar/obj_detect/pnp/map2019.png')
        size=(self.map_width,self.map_height)
        self.pic=cv.resize(self.pic,size)
        self.map_width    = 1920
        self.map_height   = 886
        self.count_width  = 10000#25000
        self.count_height = 15000
        self.radar_height = 6000 #3500

    def get_width(self):
        return self.map_width
    def get_height(self):
        return self.map_height
    def get_count_width(self):
        return self.count_width
    def get_count_height(self):
        return self.count_height
