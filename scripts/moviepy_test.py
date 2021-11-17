#!/usr/bin/env python3

from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import rospy
import glob
import numpy as np
import matplotlib.pyplot as plt
import mayavi.mlab as mlab
import moviepy.editor as mpy
import torch

from multiprocessing import Process
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V

from std_msgs.msg import String
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2


def LiDARSubscriber(data):
        rospy.loginfo(rospy.get_caller_id())
        temp_list = []

        for point in pc2.read_points(data, skip_nans=True, field_names=('x', 'y', 'z')):
            temp_list.append([point[0], point[1], point[2], 0])
        pc_data = np.array(temp_list, dtype=np.float32)
        #print(np.shape(self.pc_data), self.pc_data[0])
        #np.save('my_data3.npy', self.pc_data)
        #self.pc_data.astype(np.float32).tofile('my_data2.bin')

@mlab.animate
def anim():
    x, y = np.mgrid[0:3:1,0:3:1]
    s = mlab.surf(x, y, np.asarray(x*0.1, 'd'))

    for i in range(100000):
        s.mlab_source.scalars = np.asarray(x*0.1*(i+1), 'd')
        yield

    anim()
    mlab.show()    

if __name__ == '__main__':
    #rospy.init_node('LiDAR_listener', anonymous=True)
    #rospy.Subscriber('/points_raw', PointCloud2, LiDARSubscriber)

    

    # @mlab.animate
    # def anim():
    #     for i in range(100000):
    #         s.mlab_source.scalars = np.asarray(x*0.1*(i+1), 'd')
    #         yield

    # anim()
    # mlab.show()
    vis = Process(target=anim)
    vis.start()

    #rospy.spin()

    #node = PointCloudData()
    #node.main()

# x, y = np.mgrid[0:3:1,0:3:1]
# s = mlab.surf(x, y, np.asarray(x*0.1, 'd'))

# data = np.load('data/my_data3.npy')
# x, y, z, _ = np.transpose(data)
# value = np.ones(9407)
# print(np.shape(x), np.shape(y), np.shape(z), np.shape(value), value)
# mlab.points3d(x, y, z, scale_factor=.25)
# mlab.show()
