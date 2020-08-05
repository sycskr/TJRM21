import numpy as np
import torch

from utils import torch_utils
from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.preprocessing import non_max_suppression
from .sort.detection import Detection
from .sort.track import TrackState
from .sort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.device = None
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
        bbox_xywh : bbox_xcycwh
        confidences
        ori_img

        Returns   outputs.append(np.array([x1,y1,x2,y2,track_id, vx, vy], dtype=np.int))
        -------

        '''
        self.height, self.width = ori_img.shape[:2]

        features = self._get_features(bbox_xywh, ori_img)
        #features = np.ones((len(bbox_xywh),512), np.float32)

        #获取目标框，筛选置信度大于min_confidence的目标框
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)

        detections = [Detection(bbox_tlwh[i], conf, features[i], False) for i,conf in enumerate(confidences) if conf>self.min_confidence]


        # #添加一些上一帧的预测框
        # bbox_predict = []
        # for i,track in enumerate(self.tracker.tracks):
        #     mean = track.mean
        #     cx = mean[0] ; cy = mean[1];
        #     a = mean[2] ; h = mean[3];
        #     vx = mean[4] ; vy = mean[5];
        #
        #     #上一帧预测的目标框
        #     bbox_new = [cx+vx, cy+vy, a*h, h]
        #     # conf_new = torch.tensor([0.5],device=self.device) #注意一定要是列表形式的
        #     #新增
        #     bbox_predict.append(bbox_new)
        #
        # features_predict =  self._get_features(bbox_predict, ori_img)
        #
        # for i in range(len(features_predict)):
        #     detections.append(Detection(bbox_predict[i], 0.5, features_predict[i], True))


        # generate detections
        #采用级联分类器，对yolo筛选出来的框做了进一步的神经网络特征提取
        #t0 = torch_utils.time_synchronized()
        #print("bbox_xywh", np.array(bbox_xywh).shape)


        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        # 极大值抑制
        #print("self.nms_max_overlap : ",self.nms_max_overlap)
        indices = non_max_suppression(boxes, self.nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        #print('detections : ' , detections)
        #t1 = torch_utils.time_synchronized()
        #print('_get_features. (%.3fs)' % (t1- t0))

        tmp_tracker = self.tracker
        # update tracker 更新了所有矩阵的预测帧位置
        self.tracker.predict()

        #t2 = torch_utils.time_synchronized()
        #print('predict. (%.3fs)' %  (t2 - t1))

        # update tracker 更新了所有矩阵的预测帧位置
        self.tracker.update(detections, ori_img)

        # output bbox identities
        tl_br_id = []
        vx_vy = []
        for i,track in enumerate(self.tracker.tracks):
            #print("No.",i," is_confirmed(",track.is_confirmed() ," ",track.track_id)
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #print(track.track_id, " : ", track.mean)
            vx = track.mean[4]
            vy = track.mean[5]
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            tl_br_id.append(np.array([x1, y1, x2, y2, track_id], dtype=np.int))
            vx_vy.append(np.array([vx,vy],dtype=np.float32))

        if len(tl_br_id) > 0:
            tl_br_id = np.stack(tl_br_id,axis=0)
            vx_vy = np.stack(vx_vy, axis=0)
        return tl_br_id,vx_vy


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



        #
        # '''========================================================================================================================'''
        # # Run matching cascade. 匹配
        # matches, unmatched_tracks, unmatched_detections = \
        #     self.tracker._match(detections)
        #
        # # Update track set.
        # for track_idx, detection_idx in matches:
        #     self.tracker.tracks[track_idx].update(
        #         self.tracker.kf, detections[detection_idx])
        #
        # # 丢掉的目标 直接先使用预测框
        # for track_idx in unmatched_tracks:
        #     # 使用预测框
        #     track = self.tracker.tracks[track_idx]
        #     mean = track.mean
        #     cx = mean[0];
        #     cy = mean[1];
        #     a = mean[2];
        #     h = mean[3];
        #     vx = mean[4];
        #     vy = mean[5];
        #     # 上一帧预测的目标框
        #     bbox_new = [[cx + vx, cy + vy, a * h, h]]
        #     # bbox_new = [torch.tensor([cx + vx, cy + vy, a * h, h],device=self.device)]
        #     # 新增
        #     # bbox.append(bbox_new)
        #     # print("bbox",np.array(bbox).shape)
        #     feature = self._get_features(bbox_new, ori_img)
        #     bbox_tlwh = self._xywh_to_tlwh(bbox_new)
        #
        #     detection_new = Detection(bbox_new[0], 0.5, feature[0])
        #     # print("   track[",track_idx,"].mean : ",track.mean)
        #     if self.tracker.tracks[track_idx].state != TrackState.Temporary:
        #         self.tracker.tracks[track_idx].update_my(detection_new)  # update(self.tracker.kf, detection_new)
        #     else:
        #         self.tracker.tracks[track_idx].mark_missed()
        #     # print("   track[",track_idx,"].mean : ",self.tracker.tracks[track_idx].mean)
        #     # self.tracker.tracks[track_idx].mark_missed()
        #
        # # 新增的一些目标框,可以修改
        # for detection_idx in unmatched_detections:
        #     self.tracker._initiate_track(detections[detection_idx])
        #
        # self.tracker.tracks = [t for t in self.tracker.tracks if not t.is_deleted()]
        #
        # # Update distance metric.
        # active_targets = [t.track_id for t in self.tracker.tracks if t.is_confirmed()]
        # features, targets = [], []
        #
        # for track in self.tracker.tracks:
        #     if not track.is_confirmed():
        #         continue
        #     features += track.features
        #     targets += [track.track_id for _ in track.features]
        #     track.features = []
        #
        # self.tracker.metric.partial_fit(
        #     np.asarray(features), np.asarray(targets), active_targets)
        #
        # '''========================================================================================================================'''