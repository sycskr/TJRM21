# coding:utf-8
# vim: expandtab:ts=4:sw=4

from __future__ import absolute_import
import numpy as np
import torch

from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, metric, max_iou_distance=0.7, max_age=70, n_init=3):
        #距离方式
        self.device = None
        self.metric = metric
        #阈值
        self.max_iou_distance = max_iou_distance
        #最长存活期限
        self.max_age = max_age
        #被确认的连续命中次数
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections,ori_img):
        """Perform measurement update and track management.
        矩阵位置进行匹配
        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade. 匹配


        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        # Update track set.
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(
                self.kf, detections[detection_idx])

        #丢掉的目标
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
            # #使用预测框
            # track = self.tracks[track_idx]
            # mean = track.mean
            # cx = mean[0] ; cy = mean[1];
            # a = mean[2] ; h = mean[3];
            # vx = mean[4] ; vy = mean[5];
            # #上一帧预测的目标框
            # bbox_new = [cx+vx, cy+vy, a*h, h]
            # conf_new = torch.tensor([0.5],device=self.device) #注意一定要是列表形式的
            #
            # #新增
            # features = self._get_features(bbox_new, ori_img)
            # # features = np.ones((len(bbox_xywh),512), np.float32)
            #
            # # 获取目标框，筛选置信度大于min_confidence的目标框
            # bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
            # detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(confidences) if
            #               conf > self.min_confidence]
            #
            # print("bbox_new ",i," : ",bbox_new)
            # confidences = torch.cat((confidences,conf_new),dim = 0)

        #新增的一些目标框,可以修改
        for detection_idx in unmatched_detections:
            if(detections[detection_idx].is_predict == True):
                continue

            self._initiate_track(detections[detection_idx])

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Update distance metric.
        active_targets = [t.track_id for t in self.tracks if t.is_confirmed()]
        features, targets = [], []

        for track in self.tracks:
            if not track.is_confirmed():
                continue
            features += track.features
            targets += [track.track_id for _ in track.features]
            track.features = []
        self.metric.partial_fit(
            np.asarray(features), np.asarray(targets), active_targets)

    def _match(self, detections):
        '''
        用预测来实现跟随
        但是如果detection不准确就会跟丢，我们现在希望无论如何在小地图上都时刻显示
        所以要将预测框也要放进来
        '''
        def gated_metric(tracks, dets, track_indices, detection_indices):
            '''

            Parameters
            ----------
            tracks
            dets
            track_indices
            detection_indices

            Returns
            -------

            '''
            #这个写法很棒
            features = np.array([dets[i].feature for i in detection_indices])
            targets = np.array([tracks[i].track_id for i in track_indices])
            #print("features : \n", features) #是特征向量
            #得到特征相似矩阵
            #targets 存放的是ID , cost_matrix 存放着target与所有当前检测框（detection）的feature的余弦距离
            #cost_matrix[i,j] 表示外观空间中第 i 个轨迹和第 j 个检测之间的最小余弦距离
            #
            cost_matrix = self.metric.distance(features, targets)

            #用距离作为第一次筛选
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, dets, track_indices,
                detection_indices)

            return cost_matrix


        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Associate confirmed tracks using appearance features.
        #第一轮将detection与已经确认的目标直接匹配 ， 匹配依据 特征+距离
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(#第一个参数把函数传进去了
                gated_metric, self.metric.matching_threshold, self.max_age,
                self.tracks, detections, confirmed_tracks)

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        #第二轮 用未匹配成功的轨迹和当前的不确定的目标 进行匹配 ， 匹配依据 IOU
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]

        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections, iou_track_candidates, unmatched_detections)

        matches = matches_a + matches_b

        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))

        #if unmatched_tracks != 0 需要脑补一个轨迹

        return matches, unmatched_tracks, unmatched_detections



    def _initiate_track(self, detection):
        '''
        初始化track
        Parameters
        ----------
        detection

        Returns
        -------

        '''
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
        # 和既定序列进行匹配 修改ID
