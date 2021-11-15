#!/usr/bin/env python3

from mpl_toolkits.mplot3d import Axes3D
import rospy
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V

from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2

class PointCloudData:
    pc_data = np.array

    def __init__(self):
        rospy.Subscriber("/points_raw", PointCloud2, self.LiDARSubscriber)


    def LiDARSubscriber(self, data):
        #rospy.loginfo(rospy.get_caller_id())
        temp_list = []

        for point in pc2.read_points(data, skip_nans=True, field_names=('x', 'y', 'z')):
            temp_list.append([point[0], point[1], point[2], 0])
        self.pc_data = np.array(temp_list)
        print(self.pc_data[:, 0])
        #print(np.shape(self.pc_data), self.pc_data[0])
        #np.save('my_data.npy', self.pc_data)
        #self.pc_data.astype(np.float32).tofile('my_data.bin')


    def main(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('LiDAR_listener', anonymous=True)
    node = PointCloudData()
    node.main()
