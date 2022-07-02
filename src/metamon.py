#!/usr/bin/env python

# fisheye to panorama
# for kodak PIXPRO sp360

import rospy
from sensor_msgs.msg import Image as ImageMsg
import cv2
import numpy as np
from cv_bridge import CvBridge
import math
from numba import jit


class fish2pano():
    '''make a panorama image from a hemisphere fisheye image'''

    def __init__(self):

        # subscriver and publisher
        self.image_sub = rospy.Subscriber('/kodak/image', ImageMsg, self.callback)
        self.image_pub = rospy.Publisher('/metamon/image', ImageMsg, queue_size=1)
        self.bridge = CvBridge()
        self.data = ImageMsg()


        # image parameters
        self.input_width = 1024
        self.input_height = 1024
        # self.output_width = 1024
        # self.output_height = 1024

    def callback(self, data):
        '''callback function for subscriber'''
        rospy.loginfo_once("get image")
        # convert ROS image to CV image
        cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        # print(cv_image.shape)

        # make a panorama image
        # pano_image = self.make_pano(cv_image)
        pano_image = self.make_pano_by_scratch(cv_image)

        # convert CV image to ROS image and publish to topic
        ros_image = self.bridge.cv2_to_imgmsg(pano_image, "rgb8")
        self.image_pub.publish(ros_image)
        # cv2.imshow("pano", ros_image)

    @jit #for faster forloop
    def make_pano_by_scratch(self,cv_image):
        '''make a panorama image from a fisheye image'''

        # set params
        h = int(self.input_height / 2)
        w = int(3 * h)

        # set a panorama image
        pano_image = np.zeros((h, w, 3), np.uint8)
        # print(pano_image.shape)

        # convert CV image to numpy array
        cv_image = np.array(cv_image)
        # print(cv_image.shape)
        pano_image = np.array(pano_image)


        # make a panorama image
        for i in range(0,w):
            for j in range(0,h):
                r = j
                p = 2 * math.pi * i / w
                ix = h - r * math.cos(p)
                iy = h + r * math.sin(p)
                # if(int(ix)-1 == 0 ) or (int(iy)-1 == 0) : print(ix, iy)
                pano_image[j][i] = cv_image[int(iy-1)][int(ix-1)]

        # convert numpy array to CV image
        pano_image = cv2.cvtColor(pano_image, cv2.COLOR_BGR2RGB)

        return pano_image

# main
def main():
    '''kodak to metamon'''

    rospy.init_node("metamon_streamer")

    metamon = fish2pano()

    print ("start")
    rospy.spin()


if __name__ == "__main__":
    main()
