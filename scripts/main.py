#!/usr/bin/env python3

from pathlib import Path
from time import process_time
from mpl_toolkits.mplot3d import Axes3D
from pyquaternion import Quaternion

import ros_numpy
import rospy
import glob
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import moviepy.editor as mpy
import torch
import time

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V

from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
from sensor_msgs.msg import Image
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

import sensor_msgs.point_cloud2 as pc2


def get_annotations_indices(types, thresh, label_preds, scores):
    indexs = []
    annotation_indices = []
    for i in range(label_preds.shape[0]):
        if label_preds[i] == types:
            indexs.append(i)
    for index in indexs:
        if scores[index] >= thresh:
            annotation_indices.append(index)
    return annotation_indices  

def remove_low_score_nu(image_anno, thresh):
    img_filtered_annotations = {}
    label_preds_ = image_anno["pred_labels"].detach().cpu().numpy()
    scores_ = image_anno["pred_scores"].detach().cpu().numpy()
    
    car_indices =                  get_annotations_indices(1, 0.15, label_preds_, scores_)
    truck_indices =                get_annotations_indices(2, 0.1, label_preds_, scores_)
    construction_vehicle_indices = get_annotations_indices(3, 0.1, label_preds_, scores_)
    bus_indices =                  get_annotations_indices(4, 0.1, label_preds_, scores_)
    trailer_indices =              get_annotations_indices(5, 0.1, label_preds_, scores_)
    barrier_indices =              get_annotations_indices(6, 0.1, label_preds_, scores_)
    motorcycle_indices =           get_annotations_indices(7, 0.1, label_preds_, scores_)
    bicycle_indices =              get_annotations_indices(8, 0.1, label_preds_, scores_)
    pedestrain_indices =           get_annotations_indices(9, 0.1, label_preds_, scores_)
    traffic_cone_indices =         get_annotations_indices(10,0.1, label_preds_, scores_)
    
    for key in image_anno.keys():
        if key == 'metadata':
            continue
        img_filtered_annotations[key] = (
            image_anno[key][car_indices +
                            pedestrain_indices + 
                            bicycle_indices +
                            bus_indices +
                            construction_vehicle_indices +
                            traffic_cone_indices +
                            trailer_indices +
                            barrier_indices +
                            truck_indices
                            ])

    return img_filtered_annotations

def yaw2quaternion(yaw: float) -> Quaternion:
    return Quaternion(axis=[0,0,1], radians=yaw)


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.npy'):
        # Parent initialization
        super().__init__(dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=None, logger=logger)
        self.ext = ext

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


class ProcessROS:
    def __init__(self, cfg_file, ckpt):
        self.cfg_file = cfg_file
        self.ckpt = ckpt
        self.ext = '.npy'

    def ReadConfig(self):
        cfg_from_yaml_file(self.cfg_file, cfg)
        self.logger = common_utils.create_logger()
        self.demo_dataset = DemoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False, ext=self.ext, logger=self.logger)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.net.load_params_from_file(filename=self.ckpt, logger=self.logger, to_cpu=True)
        self.net = self.net.to(self.device).eval()

    def run(self, points):
        # row: > 9000, col: 4
        self.points = points.reshape([-1, 4])

        input_dict = {
            'points': self.points,
            'frame_id': 0,
        }

        data_dict = self.demo_dataset.prepare_data(data_dict=input_dict)
        data_dict = self.demo_dataset.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        
        torch.cuda.synchronize()
        #t = time.time()

        pred_dicts, _ = self.net.forward(data_dict)
        
        torch.cuda.synchronize()
        #print(f" pvrcnn inference cost time: {time.time() - t}")

        pred = remove_low_score_nu(pred_dicts[0], 0.45)
        boxes_lidar = pred["pred_boxes"].detach().cpu().numpy()
        scores = pred["pred_scores"].detach().cpu().numpy()
        types = pred["pred_labels"].detach().cpu().numpy()

        return scores, boxes_lidar, types

def LiDARSubscriber(data):
    #t_t = time.time()
    arr_bbox = BoundingBoxArray()

    # PointCloud2 -> points: [x, y, z, intensity(=0)]
    temp_list = []
    for point in pc2.read_points(data, skip_nans=True, field_names=('x', 'y', 'z')):
        temp_list.append([point[0], point[1], point[2], 0])
    points = np.array(temp_list, dtype=np.float32)

    scores, box_lidar, types_ = proc.run(points)

    if scores.size != 0:
        for i in range(scores.size):
            bbox = BoundingBox()
            bbox.header.frame_id = data.header.frame_id
            bbox.header.stamp = rospy.Time.now()
            q = yaw2quaternion(float(box_lidar[i][6]))
            bbox.pose.orientation.x = q[1]
            bbox.pose.orientation.y = q[2]
            bbox.pose.orientation.z = q[3]
            bbox.pose.orientation.w = q[0]           
            bbox.pose.position.x = float(box_lidar[i][0])
            bbox.pose.position.y = float(box_lidar[i][1])
            bbox.pose.position.z = float(box_lidar[i][2])
            bbox.dimensions.x = float(box_lidar[i][3])
            bbox.dimensions.y = float(box_lidar[i][4])
            bbox.dimensions.z = float(box_lidar[i][5])
            bbox.value = scores[i]
            bbox.label = int(types_[i])
            arr_bbox.boxes.append(bbox)
    
    #print("total callback time: ", time.time() - t_t)
    arr_bbox.header.frame_id = data.header.frame_id
    arr_bbox.header.stamp = data.header.stamp
    if len(arr_bbox.boxes) is not 0:
        pub_arr_bbox.publish(arr_bbox)
        arr_bbox.boxes = []
    else:
        arr_bbox.boxes = []
        pub_arr_bbox.publish(arr_bbox)
    repub_points_raw.publish(data)
    repub_image_raw.publish(re_image_raw)

def RePubImage(data):
    global re_image_raw
    re_image_raw = Image()
    re_image_raw = data
    #repub_image_raw.publish(data)

if __name__=='__main__':
    global proc

    cfg_file = '/home/msjun-msi/catkin_ws/src/3D-LiDAR-Detection/scripts/cfgs/kitti_models/voxel_rcnn_car.yaml'
    ckpt = '/home/msjun-msi/catkin_ws/src/3D-LiDAR-Detection/scripts/voxel_rcnn_car_84.54.pth'

    rospy.init_node('LiDAR_listener')
    proc = ProcessROS(cfg_file=cfg_file, ckpt=ckpt)
    proc.ReadConfig()

    rospy.Subscriber('/points_raw', PointCloud2, LiDARSubscriber)
    rospy.Subscriber('/image_raw', Image, RePubImage)
    pub_arr_bbox = rospy.Publisher('/arr_bbox', BoundingBoxArray, queue_size=1)
    repub_points_raw = rospy.Publisher('/re_points_raw', PointCloud2, queue_size=1)
    repub_image_raw = rospy.Publisher('/re_image_raw', Image, queue_size=1)

    rospy.spin()
