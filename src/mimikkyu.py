#!/usr/bim /env python

# panorama to cubemap
# why mimikkyu ? because it's a 'cu'be map

import rospy
from sensor_msgs.msg import Image as ImageMsg
import cv2
from cv_bridge import CvBridge
import numpy as np
import math
from numba import jit

class pano2cube():
    '''make a cubemap from a panorama image'''

    def __init__(self):
        #subscriber and publisher
        self.image_sub = rospy.Subscriber('/metamon/image', ImageMsg, self.callback)
        self.image_pub = rospy.Publisher('/mimikkyu/image', ImageMsg, queue_size=1)
        self.bridge = CvBridge()
        self.data = ImageMsg()

        # image parameters
        self.imput_width = 0
        self.imput_height = 0
        self.output_width = 0
        self.output_height = 0

    def callback(self, data):
        '''callback function for panorama image'''
        rospy.loginfo("mimikkyu : received image")

        # convert ROS image to CV image
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        print(cv_image.shape)

        #make a cubemap
        cube_image = self.make_cube(cv_image)

        # convert CV image to ROS image and publish to topic
        ros_image = self.bridge.cv2_to_imgmsg(cube_image, "rgb8")
        self.image_pub.publish(ros_image)
        cv2.imshow("cube", ros_image)

    @jit #for faster processing
    def make_cube(cv_image):
        ''' make a cubemap from a panorama image'''

        #set params

        #set a cubemap
        cube_image = np.zeros((, , 3), np.uint8)
        # print(cube_image.chape)

        #ここで展開図作る

        # convert CV image to numpy array
        cv_image = np.array(cv_image)
        # print(cv_image.shape)
        cube_image = np.array(cube_image)

        # make a cubemap

        # convert numpy array to CV image
        cube_image = cv2.cvtColor(cube_image, cv2.COLOR_BGR2RGB)

        return cube_image




def main():
    '''metamon to mimikkyu'''
    rospy.imit_node("mimikkyu_streamer")

    mimikkyu = pano2cube()

    print("Let's go mimikkyu!")
    rospy.spin()

if __name__ == "__main__"
    main()
