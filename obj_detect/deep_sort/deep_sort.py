import numpy as np
import torch

from utils import torch_utils
from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = 0#min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        nn_budget = 100
        metric = NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, ori_img):
        '''

        Parameters
        ----------
        bbox_xywh
        confidences
        ori_img

        Returns   outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        -------

        '''
        self.height, self.width = ori_img.shape[:2]

        # generate detections
        #采用级联分类器，对yolo筛选出来的框做了进一步的神经网络特征提取
        #这里先忽略
        #t0 = torch_utils.time_synchronized()

        #features = self._get_features(bbox_xywh, ori_img)
        features = np.ones((len(bbox_xywh),512), np.float32)#print(features.dtype)
        #print(len(bbox_xywh), " ", features.shape)

        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i,conf in enumerate(confidences) if conf>self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        #t1 = torch_utils.time_synchronized()
        #print('_get_features. (%.3fs)' % (t1- t0))

        # update tracker
        self.tracker.predict()
        #t2 = torch_utils.time_synchronized()
        #print('predict. (%.3fs)' %  (t2 - t1))

        self.tracker.update(detections)
        # output bbox identities
        outputs = []
        for i,track in enumerate(self.tracker.tracks):
            #print("No.",i," is_confirmed(",track.is_confirmed() ," ",track.track_id)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1,y1,x2,y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            outputs.append(np.array([x1,y1,x2,y2,track_id], dtype=np.int))
        if len(outputs) > 0:
            outputs = np.stack(outputs,axis=0)
        return outputs


    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
        x_center -> top_left
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()

        bbox_xywh = np.array(bbox_xywh)
        bbox_tlwh = bbox_xywh

        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2]/2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3]/2.
        return bbox_tlwh


    def _xywh_to_xyxy(self, bbox_xywh):
        x,y,w,h = bbox_xywh
        x1 = max(int(x-w/2), 0)
        x2 = min(int(x+w/2), self.width-1)
        y1 = max(int(y-h/2), 0)
        y2 = min(int(y+h/2), self.height-1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x,y,w,h = bbox_tlwh
        x1 = max(int(x),0)
        x2 = min(int(x+w),self.width-1)
        y1 = max(int(y),0)
        y2 = min(int(y+h),self.height-1)
        return x1,y1,x2,y2

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1,y1,x2,y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2-x1)
        h = int(y2-y1)
        return t,l,w,h
    
    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:

            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            #print("(",x1,",",y1,")",",""(",x2,",",y2,")")
            im = ori_img[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
            #print(im.shape)
            im_crops.append(im)


        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


