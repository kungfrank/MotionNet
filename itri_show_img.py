import argparse
import pandas as pd
import time
import os
import numpy as np
from numpy.linalg import inv
import time
import math

import matplotlib.pyplot as plt
from PIL import Image
import cv2
from matplotlib.colors import hsv_to_rgb
import matplotlib

import rospy
from sensor_msgs.msg import Image
import sys
import os
pid = os.getpid()
print('pid:', pid)

dir_path = '/mnt/Disk1/viz_result/itri_2021-05-12_14-51-57_ep20_viz4'
#dir_path1 = '/mnt/Disk1/viz_result/itri_2021-05-12_14-51-57_ep20'

win1_name = 'result'
#win2_name = 'sparse'

#cv2.namedWindow(win1_name, cv2.WINDOW_NORMAL)

def callback(data):
  radar_id = data.header.stamp.to_sec()

  img_path = dir_path+'/'+str(radar_id)+'.png'
  print(radar_id)
  #rospy.loginfo("I heard Radar %s: %s", data.header.seq, data.header.stamp.to_sec())
  #img_path2 = dir_path1+'/'+str(data.header.stamp)+'.png'

  if not os.path.isfile(img_path):
      print("Could not find cam example: {}".format(img_path))
  #if not os.path.isfile(img_path2):
  #    print("Could not find cam example: {}".format(img_path2))
  img = cv2.imread(img_path)
  #img2 = cv2.imread(img_path2)
  cv2.namedWindow(win1_name, cv2.WINDOW_NORMAL)
  cv2.imshow(win1_name, img)
  #cv2.namedWindow(win2_name, cv2.WINDOW_NORMAL)
  #cv2.imshow(win2_name, img2)

  cv2.waitKey(150)



def listener():
  rospy.init_node('listener', anonymous=True)
  rospy.Subscriber("/Navtech/Cartesian", Image, callback, queue_size=100000)
  rospy.spin()


if __name__ == '__main__':
  listener()





